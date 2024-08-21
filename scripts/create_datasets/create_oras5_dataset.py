''' Create ORAS5 dataset.

@Author  :   Jakob Schlör 
@Time    :   2023/08/09 16:18:54
@Contact :   jakob.schloer@uni-tuebingen.de
'''
# %%
# Imports and parameters
# ======================================================================================
import os, argparse#, nc_time_axis
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from tqdm import tqdm
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from s2aenso.utils import preproc

PATH = os.path.dirname(os.path.abspath(__file__))
plt.style.use(PATH + "/../../paper.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument('-path', '--dirpath', default='./',
                    type=str, help='Folder or filepath.')
parser.add_argument('-prefix', '--prefix', default='',
                    type=str, help='Folder or filepath.')
parser.add_argument('-var', '--var', default='SST',
                    type=str, help='Variable name.')
parser.add_argument('-cpus', '--n_cpus', default=8,
                    type=int, help='Number of cpus.')
parser.add_argument('-outpath', '--outpath', default='./',
                    type=str, help='Folder to store file.')
config = vars(parser.parse_args())


config['lon_range']=[130, 290]
config['lat_range']=[-31, 33]
config['grid_step']=1.0
config['climatology']='month'

rename_vars = {'sosstsst': 'sst', 'sossheig': 'ssh', 'sozotaux': 'taux'}

# %%
# Load and merge data
# ======================================================================================
print("Load and merge data")
nc_files = [os.path.join(config['dirpath'],file) for file in os.listdir(config['dirpath']) if file.endswith('.nc')]

var = config['var']
da_arr = []
for file in nc_files: 
    da_arr.append(xr.open_dataset(file)[var])

orig_da = xr.concat(da_arr, dim='time_counter', coords='all')
orig_da = orig_da.transpose('time_counter', 'y', 'x')

# %%
# Flatten and remove NaNs 
# ======================================================================================
print("Flatten and remove Nans")
n_time, n_lat, n_lon = orig_da.shape
orig_data_flat = orig_da.values.reshape((n_time, -1))
orig_lats = orig_da['nav_lat'].values.reshape((n_time, -1))
# Transform longitudes to 0 to 360 for easier cutting
orig_lons = orig_da['nav_lon'].values.reshape((n_time, -1))
orig_lons = np.where(orig_lons < 0, orig_lons + 360, orig_lons)

assert orig_lats.shape[1] == orig_lons.shape[1]
assert orig_data_flat.shape[1] == orig_lats.shape[1]

## Remove NaNs
#idx_not_nan = np.argwhere(~np.isnan(orig_lats) & ~np.isnan(orig_lons))
#orig_data_filtered = orig_data_flat[:, idx_not_nan]
#orig_lats_filtered = orig_lats[idx_not_nan]
#orig_lons_filtered = orig_lons[idx_not_nan]


# %%
# Interpolate to regular grid
# ======================================================================================
def interp_to_regular_grid(data, timeidx, orig_lats_arr, orig_lons_arr,
                           lat_grid, lon_grid, method='linear'):
    """Interpolation of dataarray to a new set of points.

    Args:
        data (np.ndarray): Data on original grid but flattened
            of shape (n_time, n_flat) or (n_flat)
        timeidx (int): Index of time in case of parralelization.
        points_origin (np.ndarray): Array of origin locations.
            of shape(n_time, n_flat) or (n_flat)  
        points_grid (np.ndarray): Array of locations to interpolate on.
            of shape (n_time, n_flat) or (n_flat)

    Returns:
        i (int): Index of time
        values_grid_flat (np.ndarray): Values on new points.
    """
    if timeidx is None:
        orig_data = data
        orig_lats, orig_lons = orig_lats_arr, orig_lons_arr
    else:
        orig_data = data[timeidx, :]
        orig_lats = orig_lats_arr[timeidx, :]
        orig_lons = orig_lons_arr[timeidx, :]

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
        orig_data_flat, timeidx, orig_lats, orig_lons,
        lat_grid, lon_grid, method='linear'
    ) for timeidx in tqdm(range(n_processes))
)

# Store interpolated dataarray 
interpolated_data = np.array([r[0] for r in results])
timeids = np.array([r[1] for r in results])
timepoints = orig_da['time_counter'].values[timeids]

interp_da = xr.DataArray(
    data=np.array(interpolated_data),
    dims=['time', 'lat', 'lon'],
    coords=dict(time=timepoints, lat=new_lats, lon=new_lons),
    name=rename_vars[orig_da.name] if orig_da.name in rename_vars.keys() else orig_da.name
)

# %%
# Change units
# ======================================================================================
if interp_da.name == 'ssh':
    interp_da *= 100
    interp_da.attrs['units'] = 'cm'

# %%
# Fair sliding approach to compute anomalies 
# ======================================================================================
print("Compute anomalies using fair sliding:", flush=True)
da_anomaly = preproc.fair_sliding_anomalies(interp_da, window_size_year=30,
                                            group='month', detrend=True)    


# %%
# Save to file 
# ======================================================================================
print("Save target to file!")
outpath = config['outpath']
if not os.path.exists(outpath):
    print(f"Create directoty {outpath}", flush=True)
    os.makedirs(outpath)
varname = da_anomaly.name
outfname =(config['outpath'] 
      + f"/{varname}_lat{'_'.join(map(str, config['lat_range']))}"
      + f"_lon{'_'.join(map(str,config['lon_range']))}_gr{config['grid_step']}.nc")

da_anomaly.to_netcdf(outfname)

# %%
