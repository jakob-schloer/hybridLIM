"""Compute PCA and store to file."""

# %%
import os
from importlib import reload

import numpy as np
import xarray as xr

from hyblim.data import eof
from hyblim.data import preproc

PATH = os.path.dirname(os.path.abspath(__file__))

# Parameters
# ======================================================================================
config = dict(
    vars=["ssta", "ssha"],
    datapaths={
        "ssta": PATH
        + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc",
        "ssha": PATH
        + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssha_lat-31_33_lon130_290_gr1.0.nc",
    },
    n_eof=[20, 10],
    outpath=PATH + "/../../data/cesm2-picontrol/pca/",
)

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
lsm = xr.open_dataset("../../data/land_sea_mask_common.nc")["lsm"]
ds = ds.where(lsm != 1, other=np.nan)

# %%
# Create PCA over the training period only
# ======================================================================================
reload(eof)
train_period = (0, int(0.8 * len(ds["time"])))
combined_eof = eof.fit_combined_eof(ds, config["n_eof"], train_period=train_period)

# %%
# Save PCA to file
# ======================================================================================
fname = eof.eof_filename(list(ds.data_vars), config["n_eof"])
eof.save_combined_eof(combined_eof, os.path.join(config["outpath"], fname))

# %%
