"""Compute the CS-LIM optimal-growth patterns and save them to file."""

import argparse
import os
from importlib import reload

import numpy as np
import pandas as pd
import xarray as xr

from hyblim.data import eof
from hyblim.data import preproc
from hyblim.model import lim

PATH = os.path.dirname(os.path.abspath(__file__))

# Parameters
# ======================================================================================
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-v", "--vars", nargs="+", default=["ssta", "ssha"], help="Variable names used.")
parser.add_argument(
    "-eof_path",
    "--eof_path",
    default=None,
    type=str,
    help="Path to precomputed EOFs. If None, defaults to the canonical file.",
)
config = vars(parser.parse_args())
config["lim_type"] = "cslim"

config["datapaths"] = {}
if "ssta" in config["vars"]:
    config["datapaths"]["ssta"] = (
        PATH
        + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc"
    )
if "ssha" in config["vars"]:
    config["datapaths"]["ssha"] = (
        PATH
        + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssha_lat-31_33_lon130_290_gr1.0.nc"
    )
config["n_eof"] = [20, 10]

if config["eof_path"] is None:
    config["eof_path"] = (
        PATH + "/../../data/cesm2-picontrol/pca/" + eof.eof_filename(list(config["datapaths"].keys()), config["n_eof"])
    )

# Optimal patterns are saved next to the LIM eval metrics so the figure scripts find them.
scorepath = PATH + f"/../../models/lim/{config['lim_type']}_{'-'.join(config['datapaths'].keys())}/metrics"
os.makedirs(scorepath, exist_ok=True)

# %%
# Load data
# ======================================================================================
print("Load data!", flush=True)
da_arr = []
for var, path in config["datapaths"].items():
    da = xr.open_dataset(path)[var]
    # Normalize data
    normalizer = preproc.Normalizer()
    da = normalizer.fit_transform(da)
    # Store normalizer as an attribute in the Dataarray for the inverse transformation
    da.attrs = normalizer.to_dict()
    da_arr.append(da)

ds = xr.merge(da_arr)

# Apply land sea mask
lsm = xr.open_dataset(PATH + "/../../data/land_sea_mask_common.nc")["lsm"]
ds = ds.where(lsm != 1, other=np.nan)

# %%
# Create PCA and fit CS-LIM on the training period
# ======================================================================================
train_period = (0, int(0.8 * len(ds["time"])))

# Create PCA (load precomputed EOFs if available, else fit on the training period)
combined_eof = eof.get_combined_eof(ds, config["n_eof"], eof_path=config["eof_path"], train_period=train_period)
data_train = combined_eof.transform(ds.isel(time=slice(*train_period)))

reload(lim)
start_month = data_train.time.dt.month[0].data
model = lim.CSLIM(tau=1)
print("Fit CS-LIM", flush=True)
model.fit(data_train.data.T, start_month, average_window=3)

# %%
# Optimal growth: leading singular vector of G(month, lag) and its evolution
# ======================================================================================
lag_arr = np.arange(1, 25, 1)
month_arr = np.arange(1, 13, 1)
n_components = combined_eof.n_components

growth_rate, optimal_init_pc, optimal_evolved_pc = [], [], []
optimal_init_map, optimal_evolved_map = [], []
for month in month_arr:
    print(f"Optimal growth for month {month}", flush=True)
    growth = np.zeros(len(lag_arr))
    z_optimal_init = np.zeros((len(lag_arr), n_components))
    z_optimal_evolved = np.zeros((len(lag_arr), n_components))
    for i, lag in enumerate(lag_arr):
        growth[i], z_optimal_init[i] = model.growth(month=month, lag=lag)
        z_optimal_evolved[i] = np.real(model.forecast_mean(z_optimal_init[i].T, month=month, lag=lag))

    growth_rate.append(xr.DataArray(data=growth, coords=dict(lag=lag_arr)))
    optimal_init_pc.append(xr.DataArray(data=z_optimal_init, coords=dict(lag=lag_arr, eof=np.arange(n_components))))
    optimal_evolved_pc.append(
        xr.DataArray(data=z_optimal_evolved, coords=dict(lag=lag_arr, eof=np.arange(n_components)))
    )

    # Reconstruct optimal initial / evolved patterns to grid space
    x_opt_init = combined_eof.reconstruction(z_optimal_init, times=lag_arr).rename_dims({"time": "lag"})
    x_opt_evolved = combined_eof.reconstruction(z_optimal_evolved, times=lag_arr).rename_dims({"time": "lag"})
    optimal_init_map.append(x_opt_init)
    optimal_evolved_map.append(x_opt_evolved)

growth_rate = xr.concat(growth_rate, dim=pd.Index(month_arr, name="month"))
optimal_init_pc = xr.concat(optimal_init_pc, dim=pd.Index(month_arr, name="month"))
optimal_evolved_pc = xr.concat(optimal_evolved_pc, dim=pd.Index(month_arr, name="month"))
optimal_init_map = xr.concat(optimal_init_map, dim=pd.Index(month_arr, name="month"))
optimal_evolved_map = xr.concat(optimal_evolved_map, dim=pd.Index(month_arr, name="month"))

# %%
# Save to file
# ======================================================================================
print(f"Save optimal patterns to {scorepath}!", flush=True)
growth_rate.to_netcdf(scorepath + "/optimal_growth_rate.nc")
optimal_init_pc.to_netcdf(scorepath + "/optimal_init_pc.nc")
optimal_evolved_pc.to_netcdf(scorepath + "/optimal_evolved_pc.nc")
optimal_init_map.to_netcdf(scorepath + "/optimal_init_map.nc")
optimal_evolved_map.to_netcdf(scorepath + "/optimal_evolved_map.nc")
