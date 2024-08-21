''' Fit LIM and hindcast the dataset.'''
# %%
import os, argparse, time
import torch
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from importlib import reload

from hyblim.model import lim
from hyblim.data import preproc, eof

PATH = os.path.dirname(os.path.abspath(__file__))

# Parameters 
# ======================================================================================
if False:
    config = dict(num_traindata = None,
                  vars=['ssta', 'ssha'],
                  lim_type='cslim')
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vars', nargs='+', default=['ssta', 'ssha'],
                        help='Variable names used.')
    parser.add_argument('-ntrain', '--num_traindata', default=None, type=int,
                        help="Number of training datapoints.")
    parser.add_argument('-lim', '--lim_type', default='cslim', type=str,
                        help="Type of lim model.")
    config = vars(parser.parse_args())

config['datapaths'] = {}
if 'ssta' in config['vars']:
    config['datapaths']['ssta'] = PATH + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc"
if 'ssha' in config['vars']:
    config['datapaths']['ssha'] = PATH + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssha_lat-31_33_lon130_290_gr1.0.nc"
config['n_eof'] = [20, 10]


if config['num_traindata'] is not None:
    outpath = (PATH + f"/../../models/lim/"
               + f"{config['lim_type']}_{'-'.join( config['datapaths'].keys() )}/"
               + f"num_traindata/n_{config['num_traindata']}/")
else:
    outpath = PATH + f"/../../models/lim/{config['lim_type']}_{'-'.join( config['datapaths'].keys() )}/"

if not os.path.exists(outpath):
    print(f"Create directoty {outpath}", flush=True)
    os.makedirs(outpath)

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

# %%
# Create PCA
# ======================================================================================
reload(eof)
eofa_lst = []
for i, var in enumerate(ds.data_vars):
    print(f"Create EOF of {var}!")
    n_components = config['n_eof'][i] if isinstance(config['n_eof'], list) else config['n_eof'] 
    eofa = eof.EmpiricalOrthogonalFunctionAnalysis(n_components)
    eofa.fit(
        ds[var].isel(time=slice(None, int(0.8*len(ds['time']))))
    )
    eofa_lst.append(eofa)
combined_eof = eof.CombinedEOF(eofa_lst, vars=list(ds.data_vars))


# %%
# Split in training and test data
# ======================================================================================
if config['num_traindata'] is None:
    train_period = (0, int(0.8*len(ds['time'])))
else:
    idx_start = np.random.randint(0, int(0.8*len(ds['time'])) - config['num_traindata'])
    train_period = (idx_start, idx_start + config['num_traindata'])
val_period = (int(0.8*len(ds['time'])), int(0.9*len(ds['time'])))
test_period = (int(0.9*len(ds['time']), len(ds['time']))) 

data = dict(
    train = combined_eof.transform(ds.isel(time=slice(*train_period))),
    val = combined_eof.transform(ds.isel(time=slice(*val_period))),
)

# %% 
# Create LIM
# ======================================================================================
reload(lim)
if config['lim_type'] == 'stlim':
    model = lim.LIM(tau=1)
    print("Fit ST-LIM", flush=True)
    model.fit(data['train'].data)
    Q = model.noise_covariance()

elif config['lim_type'] == 'cslim':
    start_month = data['train'].time.dt.month[0].data
    average_window=3
    model = lim.CSLIM(tau=1)
    print("Fit CS-LIM", flush=True)
    model.fit(data['train'].data, start_month, average_window=average_window)
    Q = model.noise_covariance()
else:
    raise ValueError("lim_type not recognized!")

# %%
# Hindcast ensemble
# ======================================================================================
def parallel_hindcast(model, lim_type, pcs, timeidx, dt, max_lag, num_members):
    """Parallelized hindcast function."""
    z_init = pcs.isel(time=timeidx).data
    t_init = pcs['time'][timeidx]

    if lim_type == 'stlim':
        integration_times, z_integration = model.euler_integration(
            z_init, dt, max_lag, num_members)
    elif lim_type == 'cslim':
        month_init = t_init.dt.month.data
        integration_times, z_integration = model.euler_integration(
            z_init, month_init, dt, max_lag, num_members)

    # Subsample only integer times
    idx_lag = np.argwhere(integration_times % 1 == 0)[:, 0]
    z_hat = z_integration[:, idx_lag, :]

    return z_hat, timeidx 


dt = 0.25
max_lag = 24
num_members = 16

for key, hindcast_pcs in data.items():
    # Hindcast data
    print(f"Ensemble hindcast {key}!", flush=True)
    n_cpus = 8 
    n_processes = len(hindcast_pcs['time'])
    results = Parallel(n_jobs=n_cpus)(
        delayed(parallel_hindcast)(model, config['lim_type'], hindcast_pcs, timeidx, dt, max_lag, num_members)
        for timeidx in tqdm(range(n_processes))
    )
    # Read results from parallel processing
    z_hindcast = np.array([r[0] for r in results])
    timeids = np.array([r[1] for r in results])
    timepoints = hindcast_pcs['time'][timeids]
    # Create xarray DataArray
    z_hindcast_ensemble = xr.DataArray(
        data=z_hindcast,
        coords=dict(time=timepoints, 
                    member=np.arange(num_members),
                    lag=np.arange(0, max_lag+1, 1, dtype=int),
                    eof=hindcast_pcs['eof']),
        name='z',
    )

    # Save data
    print("Save data!", flush=True)
    z_hindcast_ensemble.to_netcdf(outpath 
                         + f"/{config['lim_type']}_hindcast"
                         + f"_{'-'.join(ds.data_vars)}"
                         + f"_eof{config['n_eof']}_{key}.nc")

