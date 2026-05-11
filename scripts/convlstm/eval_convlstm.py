"""Helper functions for SwinLSTM model.

@Author  :   Jakob Schlör
@Time    :   2023/08/23 15:15:35
@Contact :   jakob.schloer@uni-tuebingen.de
"""

import argparse
import glob
import json
# %%
import os

import numpy as np
import torch
import xarray as xr
from tqdm import tqdm

from hyblim.data import dataloader
from hyblim.data import preproc
from hyblim.model import convlstm
from hyblim.utils import enso
from hyblim.utils import metric

PATH = os.path.dirname(os.path.abspath(__file__))
id = int(os.environ.get("SLURM_LOCALID", 0))
device = torch.device("cuda", id) if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}", flush=True)


def hindcast_evaluation(loader, model, device, history=4, horizon=24):
    """Perform hindcast of ConvLSTM model and compute verification metrics.

    Args:
        loader (torch.utils.data.DataLoader): Dataloader for evaluation
        model (torch.nn.Module): Trained ConvLSTM model
        device (torch.device): Device to run model on
        history (int): Number of lagged time steps
        horizon (int): Number of forecast time steps

    Returns:
        verification_per_gridpoint (dict): Verification metrics per gridpoint
        verification_per_time (dict): Verification metrics per time.
        nino_indices (dict): Nino indices for forecast and target
    """
    ds = loader.dataset.data
    normalizer = {var: preproc.normalizer_from_dict(ds[var].attrs) for var in ds.data_vars}
    num_members = model.model.num_tails if hasattr(model, "model") else model.num_tails

    forecasts, targets = [], []
    with torch.no_grad():
        for i, (sample, aux) in tqdm(enumerate(loader)):
            n_batch, n_vars, n_time, n_lat, n_lon = sample.shape
            unused = n_time - history - horizon
            x_input, x_target, _ = sample.to(device).split([history, horizon, unused], dim=2)
            context, _ = aux["month"].to(device, dtype=torch.long).split([history + horizon, unused], dim=-1)
            # (batch, member, vars, lag, lat, lon)
            x_pred = model(x_input, context=context)

            # Convert to xarray
            dates_batch = ds.time[aux["idx"][:, history].numpy().astype(int)]
            dims = ["time", "member", "lag", "lat", "lon"]
            coords = {
                "time": dates_batch,
                "member": np.arange(1, num_members + 1),
                "lag": np.arange(1, horizon + 1),
                "lat": ds.lat,
                "lon": ds.lon,
            }
            xr_pred = metric.torch_to_xarray(x_pred.permute(2, 0, 1, 3, 4, 5), list(ds.data_vars), dims, **coords)
            # Unnormalize
            xr_pred = xr.merge([normalizer[var].inverse_transform(xr_pred[var]) for var in xr_pred.data_vars])

            dims = ["time", "lag", "lat", "lon"]
            coords = {"time": dates_batch, "lag": np.arange(1, horizon + 1), "lat": ds.lat, "lon": ds.lon}
            xr_target = metric.torch_to_xarray(x_target.permute(1, 0, 2, 3, 4), list(ds.data_vars), dims, **coords)
            # Unnormalize
            xr_target = xr.merge([normalizer[var].inverse_transform(xr_target[var]) for var in xr_target.data_vars])
            forecasts.append(xr_pred)
            targets.append(xr_target)

    forecasts = xr.concat(forecasts, dim="time")
    targets = xr.concat(targets, dim="time")

    # Metrics
    nino_indices = {
        "frcst": enso.get_nino_indices(forecasts["ssta"]),
        "target": enso.get_nino_indices(targets["ssta"]),
    }

    verification_per_gridpoint = metric.verification_metrics_per_gridpoint(
        targets, forecasts.mean(dim="member"), forecasts.std(dim="member"), forecasts.dims["member"]
    )
    verification_per_time = metric.verification_metrics_per_time(
        targets, forecasts.mean(dim="member"), forecasts.std(dim="member"), forecasts.dims["member"]
    )

    return verification_per_gridpoint, verification_per_time, nino_indices


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("-datasplit", "--datasplit", type=str, default="test", help="Datasplit to evaluate")
    params = vars(parser.parse_args())
    return params


# %%
if __name__ == "__main__":
    # Specify parameters
    params = argument_parser()

    with open(params["model_path"] + "/config.json", "r") as f:
        config = json.load(f)

    config["batch_size"] = 16

    # Load data
    ds, datasets, dataloaders = dataloader.load_stdata(**config)
    config["input_dim"] = len(list(ds.data_vars))

    # Define and load model
    condition_num = 12 if config["film"] else -1
    model = convlstm.EncDecSwinLSTM(
        input_dim=config["input_dim"],
        num_channels=config["num_channels"],
        output_dim=config["input_dim"],
        patch_size=(4, 4),
        num_layers=config["num_layers"],
        num_conditions=condition_num,
        num_tails=config["members"],
    ).to(device)

    # Load model with best loss (Lightning checkpoint: weights stored with "model." prefix)
    ckpt_file = glob.glob(params["model_path"] + "/best-checkpoint-epoch*")[0]
    checkpoint = torch.load(ckpt_file, map_location=device)
    state_dict = {k.replace("model.", "", 1): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state_dict)
    model.eval()

    # Evaluate model
    verification_per_gridpoint, verification_per_time, nino_indices = hindcast_evaluation(
        dataloaders[params["datasplit"]], model, device, history=4, horizon=24
    )

    # Save metrics to file
    scorepath = params["model_path"] + "/metrics"
    print("Save metrics to file!", flush=True)
    if not os.path.exists(scorepath):
        os.makedirs(scorepath)

    for key, score in verification_per_gridpoint.items():
        score.to_netcdf(scorepath + f"/gridscore_{key}_{params['datasplit']}.nc")
    for key, score in verification_per_time.items():
        score.to_netcdf(scorepath + f"/timescore_{key}_{params['datasplit']}.nc")
    for key, nino_idx in nino_indices.items():
        nino_idx.to_netcdf(scorepath + f"/nino_{key}_{params['datasplit']}.nc")

    # %%
