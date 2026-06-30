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

from hyblim.data import dataloader
from hyblim.data import eof
from hyblim.data import preproc
from hyblim.model import lstm
from hyblim.utils import eval
from hyblim.utils import metric

reload(metric)

PATH = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def hindcast(model, dataloader, normalizer_pca):
    targets, frcsts, time_ids = [], [], []
    with torch.no_grad():
        for lim_input, target, aux in dataloader:
            x_input, x_target = lim_input.to(DEVICE), target.to(DEVICE)
            context = aux["month"].to(DEVICE, dtype=torch.long)
            # Prediction
            x_ensemble = model(x_input, context)

            targets.append(x_target.cpu())
            time_ids.append(aux["idx"])
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
    """Save the corrected (hybrid) hindcast in PC space.

    Mirrors the CS-LIM hindcast file so figures can reconstruct grid-space
    LIM-LSTM forecasts without re-running the model.

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
    outpath = os.path.join(modelpath, f"limlstm_hindcast_{vars_str}_eof{eof_str}_{datasplit}.nc")
    z_da.to_dataset().to_netcdf(outpath)
    print(f"Saved hybrid hindcast to {outpath}", flush=True)
    return outpath


def perform_hindcast_evaluation(
    model: torch.nn.Module,
    ds: xr.Dataset,
    dataloaders: torch.utils.data.DataLoader,
    datasplit: str,
    scaler_pca: preproc.Normalizer,
    combined_eof: eof.CombinedEOF,
    lag_arr: list,
    scorepath: str,
) -> None:
    """Perform hindcast and compute verification metrics for LSTM model.

    Args:
        model (torch.nn.Module): LSTM model
        ds (xr.Dataset): Dataset
        dataloaders (dict): Dictionary of dataloaders.
        datasplit (str): Datasplit to evaluate, i.e. 'train', 'val', 'test'
        scaler_pca (preproc.Normalizer): Normalizer for PCA space
        combined_eof (eof.CombinedEOF): Combined EOF object
        lag_arr (list): List of lags to compute metrics for
        scorepath (str): Path to save metrics
    """
    # Hindcast in latent space
    z_hindcast, time_idx = hindcast(model, dataloaders[datasplit], scaler_pca)

    # Save the corrected hindcast (PC space) for downstream grid-space figures.
    # time_idx[:, 0] is the lag-1 valid time, so the initialization is one month
    # earlier; key the hindcast by that initialization time.
    times = dataloaders[datasplit].dataset.data["time"].data
    init_times = times[time_idx[:, 0] - 1]
    save_latent_hindcast(
        z_hindcast["frcst"],
        init_times,
        combined_eof,
        os.path.dirname(scorepath.rstrip("/")),
        datasplit,
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
        score.to_netcdf(scorepath + f"/gridscore_{key}_{datasplit}.nc")
    for key, score in verification_per_time.items():
        score.to_netcdf(scorepath + f"/timescore_{key}_{datasplit}.nc")
    for key, nino_idx in nino_indices.items():
        nino_idx.to_netcdf(scorepath + f"/nino_{key}_{datasplit}.nc")

    return None


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("-datasplit", "--datasplit", type=str, default="test", help="Datasplit to evaluate")
    parser.add_argument(
        "-lags", "--lags", nargs="+", default=[1, 3, 6, 9, 12, 15, 18, 21, 24], help="Lags to compute metrics for."
    )
    params = vars(parser.parse_args())
    return params


def build_model_and_data(model_path):
    """Load the LIM-LSTM model and its LIM-ensemble dataloaders from `model_path`.

    Returns (model, ds, dataloaders, combined_eof, normalizer_pca).
    """
    with open(model_path + "/config.json", "r") as f:
        config = json.load(f)

    # Replace stored paths with this checkout's locations.
    config["path"] = PATH + "/../../models/limlstm/"
    config["postfix"] = ""
    config["evaluate"] = True
    config["datapaths"] = {
        "ssta": PATH
        + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc",
        "ssha": PATH
        + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssha_lat-31_33_lon130_290_gr1.0.nc",
    }
    config["lsm_path"] = PATH + "/../../data/land_sea_mask_common.nc"
    config["lim_path"] = PATH + "/../../models/lim/cslim_ssta-ssha/cslim_hindcast_ssta-ssha_eof20-10"

    lim_hindcast = {
        key: xr.open_dataset(config["lim_path"] + f"_{key}.nc")["z"].sel(lag=slice(1, None))
        for key in ["train", "val", "test"]
    }
    ds, _, dataloaders, combined_eof, normalizer_pca = dataloader.load_pcdata_lim_ensemble(lim_hindcast, **config)

    num_condition = 12 if config["film"] else -1
    model = lstm.ResidualLSTM(
        input_dim=combined_eof.n_components,
        hidden_dim=config["hidden_dim"],
        num_conditions=num_condition,
        num_layers=config["layers"],
        T_max=config["chrono"],
    )
    checkpoint = torch.load(model_path + "/final_checkpoint.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    return model, ds, dataloaders, combined_eof, normalizer_pca


def main(params):
    model, ds, dataloaders, combined_eof, normalizer_pca = build_model_and_data(params["model_path"])
    lag_arr = [int(lag) for lag in params["lags"]]
    scorepath = params["model_path"] + "/metrics"
    perform_hindcast_evaluation(
        model, ds, dataloaders, params["datasplit"], normalizer_pca, combined_eof, lag_arr, scorepath
    )


if __name__ == "__main__":
    main(argument_parser())

# %%
