"""Perfrom hindcast of LSTM model and compute verification metrics.

@Author  :   Jakob Schlör
@Time    :   2023/08/23 14:40:23
@Contact :   jakob.schloer@uni-tuebingen.de
"""

# %%
import argparse
import json
import os
from importlib import reload

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from hyblim.data import dataloader
from hyblim.data import eof
from hyblim.data import preproc
from hyblim.model import lstm
from hyblim.utils import eval
from hyblim.utils import metric

reload(metric)

PATH = os.path.dirname(os.path.abspath(__file__))


def hindcast(model, dataloader, normalizer_pca, device, history, horizon):
    targets, frcsts, time_ids = [], [], []
    with torch.no_grad():
        unused = dataloader.dataset.n_timesteps - history - horizon
        for sample, aux in tqdm(dataloader):
            x_input, x_target, _ = sample.to(device).split([history, horizon, unused], dim=1)
            context, _ = aux["month"].to(device=device, dtype=torch.long).split([history + horizon, unused], dim=1)
            x_ensemble = model(x_input, context)

            targets.append(x_target.cpu())
            time_ids.append(aux["idx"][:, history:])
            frcsts.append(x_ensemble.cpu())

    time_idx = torch.cat(time_ids, dim=0).to(dtype=int).numpy()
    z_hindcast = {}
    z_hindcast["target"] = torch.cat(targets, dim=0).unsqueeze(dim=1).numpy()
    z_hindcast["frcst"] = torch.cat(frcsts, dim=0).numpy()

    # Unnormalize in PCA space
    for key, z_norm in z_hindcast.items():
        n_time, n_member, n_lag, n_components = z_norm.shape
        # Convert to xarray with eof dimension to unnormalize each eof seperately
        z_norm = xr.DataArray(
            z_norm,
            coords=dict(
                n_time=np.arange(n_time),
                n_member=np.arange(n_member),
                lag=np.arange(1, n_lag + 1),
                eof=np.arange(1, n_components + 1),
            ),
        )
        z = np.zeros((n_time, n_member, n_lag, n_components))
        for i in range(n_member):
            for j in range(n_lag):
                z[:, i, j, :] = normalizer_pca.inverse_transform(z_norm[:, i, j, :])
        z_hindcast[key] = z

    z_hindcast["target"] = z_hindcast["target"].squeeze()
    return z_hindcast, time_idx


def save_latent_hindcast(z_frcst, init_times, combined_eof, modelpath, datasplit) -> str:
    """Save the LSTM hindcast in PC space.

    Mirrors the CS-LIM / LIM-LSTM hindcast files so figures can reconstruct
    grid-space LSTM forecasts without re-running the model.

    Args:
        z_frcst (np.ndarray): Forecast PCs (time, member, lag, eof).
        init_times (np.ndarray): Initialization time of each sample.
        combined_eof (eof.CombinedEOF): EOF used (for the filename + eof coord).
        modelpath (str): Model root folder to write into.
        datasplit (str): 'train' | 'val' | 'test'.

    Returns:
        str: Path to the saved netCDF file.
    """
    n_time, n_member, n_lag, n_comp = z_frcst.shape
    z_da = xr.DataArray(
        z_frcst,
        dims=["time", "member", "lag", "eof"],
        coords=dict(
            time=init_times,
            member=np.arange(n_member),
            lag=np.arange(1, n_lag + 1),
            eof=np.arange(1, n_comp + 1),
        ),
        name="z",
    )
    vars_str = "-".join(combined_eof.vars)
    eof_str = "-".join(str(e.n_components) for e in combined_eof.eofa_lst)
    outpath = os.path.join(modelpath, f"lstm_hindcast_{vars_str}_eof{eof_str}_{datasplit}.nc")
    z_da.to_dataset().to_netcdf(outpath)
    print(f"Saved LSTM hindcast to {outpath}", flush=True)
    return outpath


def perform_hindcast_evaluation(
    model: torch.nn.Module,
    checkpoint: dict,
    ds: xr.Dataset,
    dataloader: torch.utils.data.DataLoader,
    scaler_pca: preproc.Normalizer,
    combined_eof: eof.CombinedEOF,
    lag_arr: list,
    scorepath: str,
) -> None:
    """Perform hindcast and compute verification metrics for LSTM model.

    Args:
        model (torch.nn.Module): LSTM model
        checkpoint (dict): Checkpoint of model
        ds (xr.Dataset): Dataset
        dataloader (torch.utils.data.DataLoader): Dataloader
        scaler_pca (preproc.Normalizer): Normalizer for PCA space
        combined_eof (eof.CombinedEOF): Combined EOF object
        lag_arr (list): List of lags to compute metrics for
        scorepath (str): Path to save metrics
    """
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _ = model.to(device)

    # Hindcast in latent space
    z_hindcast, time_idx = hindcast(model, dataloader, scaler_pca, device, history=4, horizon=24)

    # Save the hindcast (PC space) for downstream grid-space figures.
    # time_idx[:, 0] is the lag-1 valid time, so the initialization is one month
    # earlier; key the hindcast by that initialization time.
    times = dataloader.dataset.data["time"].data
    init_times = times[time_idx[:, 0] - 1]
    save_latent_hindcast(
        z_hindcast["frcst"],
        init_times,
        combined_eof,
        os.path.dirname(scorepath.rstrip("/")),
        "test",
    )

    # Extended PCA with 300 components
    n_components_full = 300
    eofa_list = []
    for i, var in enumerate(ds.data_vars):
        print(f"Create extended EOF of {var}!", flush=True)
        eofa = eof.EmpiricalOrthogonalFunctionAnalysis(
            n_components=n_components_full,
        )
        eofa.fit(ds[var])
        eofa_list.append(eofa)
    extended_eof = eof.CombinedEOF(eofa_list, vars=list(ds.data_vars))

    # Verification metrics
    ds_target = ds.sel(time=times)
    verification_per_gridpoint, verification_per_time, nino_indices = eval.latent_evaluation(
        z_hindcast["frcst"], time_idx, times, combined_eof, ds_target, lag_arr, extended_eof
    )

    # Save metrics to file
    print("Save metrics to file!", flush=True)
    if not os.path.exists(scorepath):
        os.makedirs(scorepath)

    for key, score in verification_per_gridpoint.items():
        score.to_netcdf(scorepath + f"/gridscore_{key}_test.nc")
    for key, score in verification_per_time.items():
        score.to_netcdf(scorepath + f"/timescore_{key}_test.nc")
    for key, nino_idx in nino_indices.items():
        nino_idx.to_netcdf(scorepath + f"/nino_{key}_test.nc")

    return None


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("-data", "--datasplit", type=str, default="test", help="Datasplit to evaluate")
    parser.add_argument(
        "-lags", "--lags", nargs="+", default=[1, 3, 6, 9, 12, 15, 18, 21, 24], help="Lags to compute metrics for."
    )
    params = vars(parser.parse_args())
    return params


if __name__ == "__main__":
    # Specify parameters
    params = argument_parser()
    with open(params["model_path"] + "/config.json", "r") as f:
        config = json.load(f)

    # Load data
    ds, datasets, dataloaders, combined_eof, scaler_pca = dataloader.load_pcdata(**config)

    # Define and load model
    condition_dim = 12 if config["film"] else None
    model = lstm.EncDecLSTM(
        input_dim=combined_eof.n_components,
        hidden_dim=config["hidden_dim"],
        num_layers=config["layers"],
        num_conditions=condition_dim,
        num_tails=config["members"],
        T_max=config["chrono"],
    )

    # Load model with best loss
    checkpoint = torch.load(params["model_path"] + "/min_checkpoint.pt")

    lag_arr = [int(lag) for lag in params["lags"]]
    scorepath = params["model_path"] + "/metrics"
    perform_hindcast_evaluation(
        model, checkpoint, ds, dataloaders[params["datasplit"]], scaler_pca, combined_eof, lag_arr, scorepath
    )


# %%
