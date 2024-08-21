''' Create dataset from CESM2 piControl run.
@Author  :   Jakob Schlör 
@Time    :   2023/08/09 16:18:54
@Contact :   jakob.schloer@uni-tuebingen.de
'''
# %%
# Imports and parameters
# ======================================================================================
import os, argparse, nc_time_axis
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed

from hyblim.utils import preproc

PATH = os.path.dirname(os.path.abspath(__file__))
plt.style.use(PATH + "/../../paper.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--dirpath', default='./',
                    type=str, help='Folder or filepath.')
parser.add_argument('-prefix', '--prefix', default='',
                    type=str, help='Folder or filepath.')
parser.add_argument('-lsm', '--lsm', default='sftlf_fx_CESM2_piControl_r1i1p1f1.nc',
                    type=str, help='Filepath.')
parser.add_argument('-var', '--var', default='SST',
                    type=str, help='Variable name.')
parser.add_argument('-cpus', '--n_cpus', default=8,
                    type=int, help='Number of cpus.')
parser.add_argument('-outpath', '--outpath', default='./../../data/cesm2-picontrol/',
                    type=str, help='Output filename.')
config = vars(parser.parse_args())

config['lon_range']=[130, 290]
config['lat_range']=[-31, 33]
config['grid_step']=1.0
config['climatology']='month'

rename_vars = {'SST': 'sst', 'SSH': 'ssh', 'TAUX': 'taux'}

# %%
# Load and merge data
# ======================================================================================
print("Load and merge data")
var = config['var']
print(os.path.join(config['dirpath'], config['prefix']) + "*.nc", flush=True)
orig_ds = xr.open_mfdataset(os.path.join(config['dirpath'], config['prefix']) + "*.nc", 
                            chunks={'time': 240},
                            combine='by_coords')
orig_da = orig_ds[var]

if var == 'SST':
    orig_da = orig_da.isel(z_t=0)

# %%
# Flatten and remove NaNs 
# ======================================================================================
n_time, n_lat, n_lon = orig_da.shape
orig_data_flat = orig_da.values.reshape((n_time, -1))
orig_lats = orig_da['TLAT'].values.ravel()
orig_lons = orig_da['TLONG'].values.ravel()
# Remove NaNs
idx_not_nan = np.argwhere(~np.isnan(orig_lats) & ~np.isnan(orig_lons))
orig_data_filtered = orig_data_flat[:, idx_not_nan]
orig_lats_filtered = orig_lats[idx_not_nan]
orig_lons_filtered = orig_lons[idx_not_nan]


# %%
# Interpolate to regular grid
# ======================================================================================
def interp_to_regular_grid(data, timeidx, orig_lats, orig_lons,
                           lat_grid, lon_grid, method='linear'):
    """Interpolation of dataarray to a new set of points.

    Args:
        data (np.ndarray): Data on original grid but flattened
            of shape (n_time, n_flat) or (n_flat)
        timeidx (int): Index of time in case of parralelization.
        points_origin (np.ndarray): Array of origin locations.
            of shape (n_flat) 
        points_grid (np.ndarray): Array of locations to interpolate on.
            of shape (n_flat) 

    Returns:
        i (int): Index of time
        values_grid_flat (np.ndarray): Values on new points.
    """
    if timeidx is None:
        orig_data = data
    else:
        orig_data = data[timeidx, :]

    values_grid = griddata(
        (orig_lats.ravel(), orig_lons.ravel()),
        orig_data.ravel(),
        (lat_grid, lon_grid),
        method=method
    )
    return values_grid, timeidx

# Create array of new grid points 
new_lats = np.arange(*config['lat_range'], config['grid_step'])
new_lons = np.arange(*config['lon_range'], config['grid_step'])
lon_grid, lat_grid = np.meshgrid(new_lons, new_lats)

# Interpolate each time step in parallel
print("Interpolate to regular grid")
n_processes = n_time
results = Parallel(n_jobs=config['n_cpus'])(
    delayed(interp_to_regular_grid)(
        orig_data_filtered, timeidx, orig_lats_filtered, orig_lons_filtered,
        lat_grid, lon_grid, method='linear'
    ) for timeidx in tqdm(range(n_processes))
)

# Store interpolated dataarray 
interpolated_data = np.array([r[0] for r in results])
timeids = np.array([r[1] for r in results])
timepoints = orig_da['time'].values[timeids]

interp_da = xr.DataArray(
    data=np.array(interpolated_data),
    dims=['time', 'lat', 'lon'],
    coords=dict(time=timepoints, lat=new_lats, lon=new_lons),
    name=rename_vars[orig_da.name] if orig_da.name in rename_vars.keys() else orig_da.name
)
# %%
# Detrend and compute anomalies 
# ======================================================================================
print("Compute anomalies:", flush=True)
da_detrend = preproc.detrend(interp_da, dim='time', deg=1, startyear=None)
da_anomaly = preproc.compute_anomalies(interp_da, group='month')
da_anomaly.name = f"{interp_da.name}a"

# %%
# Save to file 
# ======================================================================================
print("Save target to file!")
outpath = config['outpath']
if not os.path.exists(outpath):
    print(f"Create directoty {outpath}", flush=True)
    os.makedirs(outpath)
varname = da_anomaly.name
normalize = f"_norm-{config['normalization']}" if config['normalization'] is not None else ""
outfname =(os.path.join(config['outpath'], config['prefix'])  
      + f"{varname}_lat{'_'.join(map(str, config['lat_range']))}"
      + f"_lon{'_'.join(map(str,config['lon_range']))}_gr{config['grid_step']}"
      + f"{normalize}.nc")

da_anomaly.to_netcdf(outfname)

# %%
