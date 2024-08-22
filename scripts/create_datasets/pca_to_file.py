''' Compute PCA and store to file.'''
# %%
import os
import numpy as np
import xarray as xr
import joblib 
from importlib import reload
from hyblim.data import preproc, eof

PATH = os.path.dirname(os.path.abspath(__file__))

# Parameters 
# ======================================================================================
config = dict(
    vars=['ssta', 'ssha'],
    datapaths = {
        'ssta': PATH + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc",
        'ssha': PATH + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssha_lat-31_33_lon130_290_gr1.0.nc"
    },
    n_eof = [20, 10], 
    outpath = PATH + "/../../data/cesm2-picontrol/pca/",
)

# %%
# Load data
# ======================================================================================
print("Load data!", flush=True)
da_arr = []
for var, path in config['datapaths'].items():
    da = xr.open_dataset(path)[var]
    # Normalize data 
    normalizer = preproc.Normalizer()
    da = normalizer.fit_transform(da)
    # Store normalizer as an attribute in the Dataarray for the inverse transformation
    da.attrs = normalizer.to_dict()
    da_arr.append(da)

ds = xr.merge(da_arr)

# Apply land sea mask
lsm = xr.open_dataset("../../data/land_sea_mask_common.nc")['lsm']
ds = ds.where(lsm!=1, other=np.nan)

# Select training period
ds_fit = ds.isel(time=slice(None, int(0.8*len(ds['time']))))

# %%
# Create PCA
# ======================================================================================
reload(eof)
eofa_lst = []
for i, var in enumerate(ds_fit.data_vars):
    print(f"Create EOF of {var}!")
    n_components = config['n_eof'][i] if isinstance(config['n_eof'], list) else config['n_eof'] 
    eofa = eof.EmpiricalOrthogonalFunctionAnalysis(n_components)
    eofa.fit(ds_fit[var])
    eofa_lst.append(eofa)
combined_eof = eof.CombinedEOF(eofa_lst, vars=list(ds.data_vars))

# %%
# Save PCA to file
# ======================================================================================
if not os.path.exists(config['outpath']):
    print(f"Create directory {config['outpath']}", flush=True)
    os.makedirs(config['outpath'])

# Create filename, e.g. ssta_n20_ssha_n10_pca.pkl
fname = "pca_" + "_".join([f"{var}_n{n}" for var, n in zip(ds.data_vars, config['n_eof'])]) + ".pkl"
joblib.dump(combined_eof, os.path.join(config['outpath'], fname))

# %%
