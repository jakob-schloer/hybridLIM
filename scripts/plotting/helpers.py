"""Helper functions for ML diagnostics.
@author: jakob.schloer@ecmwf.int 
"""
import os, glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata
from pyshtools.expand import SHGLQ
from pyshtools.expand import SHExpandGLQ

def get_staged_exp(datadir, pattern):
    """Get files matching the pattern in the datadir.

    Args:
    datadir: str
        Directory where the hindcast files are stored.
    pattern: str
        Pattern to match the hindcast files. 
        e.g. 'ifis_ens_0_t2m_*' will match all files starting with 'ifis_ens_0_t2m_'.
    
    Returns:
    ds: xarray.Dataset
        Concatenated dataset with init_time as the new dimension
    """
    full_pattern = os.path.join(datadir, pattern)
    # Find all files matching the pattern
    files = glob.glob(full_pattern)
    # Extract dates from filenames and convert to datetime
    init_dates = [pd.to_datetime(file.split('_')[-1][:8]) for file in files]

    ds_list = []
    for f in files:
        tmp = xr.open_dataset(f)
        # Replace time dimension with lag dimension i.e. 0 to len(time) 
        tmp = tmp.rename({'time': 'lead'})
        tmp['lead'] = np.arange(1, len(tmp.lead)+1)
        ds_list.append(tmp)

    return xr.concat(ds_list, dim=pd.Index(init_dates, name='init_time')).sortby('init_time')    

# =============================================================================
# Metrics spatial average
# =============================================================================
def amean_astd(da):
    """Calculate the area weighted standard deviation of a DataArray.

    Args:
    da: xarray.DataArray
        DataArray for which the area weighted standard deviation is to be calculated.
    
    Returns:
    (xarray.DataArray, xarray.DataArray)
        Area weighted mean of the input DataArray.
        Area weighted standard deviation of the input DataArray.
    """
    dims=('latitude', 'longitude')
    weights = np.cos(np.deg2rad(da.latitude))
    weighted_mean = (da * weights).sum(dim=dims) / (weights.sum(dim='latitude')*len(da.longitude))
    weighted_variance = (weights * (da - weighted_mean)**2).sum(dim=dims) / (weights.sum(dim='latitude')*len(da.longitude))

    # Return the square root of the weighted variance
    return weighted_mean, np.sqrt(weighted_variance)

def amean_bias(frcst: xr.DataArray, target: xr.DataArray):
    """Calculate the area weighted bias between two DataArrays.

    Args:
    frcst: xarray.DataArray
        Forecast DataArray.
    target: xarray.DataArray
        Target DataArray.
    
    Returns:
    xarray.DataArray
        Area weighted bias between the forecast and target DataArrays.
    """
    dims=('latitude', 'longitude')
    weights = np.cos(np.deg2rad(frcst.latitude))
    bias = ((frcst - target) * weights).sum(dim=dims) / (weights.sum(dim='latitude')*len(frcst.longitude))
    return bias

def amean_rmse(frcst: xr.DataArray, target: xr.DataArray):
    """Calculate the area weighted root mean square error between two DataArrays.

    Args:
    frcst: xarray.DataArray
        Forecast DataArray.
    target: xarray.DataArray
        Target DataArray.
    
    Returns:
    xarray.DataArray
        Area weighted root mean square error between the forecast and target DataArrays.
    """
    dims=('latitude', 'longitude')
    weights = np.cos(np.deg2rad(frcst.latitude))
    rmse = np.sqrt(((frcst - target)**2 * weights).sum(dim=dims) / (weights.sum(dim='latitude')*len(frcst.longitude)))
    return rmse

def pattern_correlation(frcst: xr.DataArray, target: xr.DataArray):
    """Calculate the spatial correlation between two DataArrays.

    Args:
    frcst: xarray.DataArray
        Forecast DataArray.
    target: xarray.DataArray
        Target DataArray.
    
    Returns:
    xarray.DataArray
        Spatial correlation between the forecast and target DataArrays.
    """
    dims=('latitude', 'longitude')
    # Spatial anomalies
    frcst_anom = frcst - frcst.mean(dim=dims) 
    target_anom = target - target.mean(dim=dims)
    corr = xr.corr(target_anom, frcst_anom, dim=dims)
    return corr


def compute_spectra(field: np.ndarray) -> np.ndarray:
    """Compute spectral variability of a field by wavenumber.

    Adapted from: [aifs-mono](https://github.com/ecmwf-lab/aifs-mono/blob/86eabcfe363d3ab00a6b3ecbb532ea41db3083ab/aifs/diagnostics/plots.py#L11)

    Parameters
    ----------
    field : np.ndarray
        lat lon field to calculate the spectra of

    Returns
    -------
    np.ndarray
        spectra of field by wavenumber
    """
    field = np.array(field)

    # compute real and imaginary parts of power spectra of field
    lmax = field.shape[0] - 1  # maximum degree of expansion
    zero_w = SHGLQ(lmax)
    coeffs_field = SHExpandGLQ(field, w=zero_w[1], zero=zero_w[0])

    # Re**2 + Im**2
    coeff_amp = coeffs_field[0, :, :] ** 2 + coeffs_field[1, :, :] ** 2

    # sum over meridional direction
    spectra = np.sum(coeff_amp, axis=0)

    return spectra


class EquirectangularProjection:
    """Class to convert lat/lon coordinates to Equirectangular coordinates."""

    def __init__(self) -> None:
        self.x_offset = 0.0
        self.y_offset = 0.0

    def __call__(self, lon, lat):
        lon_rad = np.radians(lon)
        lat_rad = np.radians(lat)
        x = [v - 2 * np.pi if v > np.pi else v for v in lon_rad]
        y = lat_rad
        return x, y

    def inverse(self, x, y):
        lon = np.degrees(x)
        lat = np.degrees(y)
        return lon, lat

def compute_equidistant_spectra(
    da_target: xr.DataArray,
    da_frcst: xr.DataArray
):
    """Get power spectrum.


    Args:
    da_target: xr.DataArray
        Target data with dimensions (latitude, longitude).
    da_frcst: xr.DataArray
        Forecast data with dimensions (latitude, longitude).

    Returns:
    (np.ndarray, np.ndarray)
        Amplitude of spectrum (y_true), Amplitude of spectrum (y_pred)
    """
    # Interpolate the data to a regular grid
    meshlat, meshlon = np.meshgrid(da_target.latitude, da_target.longitude)
    latlons = np.stack([meshlat.flatten(), meshlon.flatten()], axis=1)
    y_true = da_target.values.flatten()
    y_pred = da_frcst.values.flatten()

    pc = EquirectangularProjection()
    lat, lon = latlons[:, 0], latlons[:, 1]
    pc_lon, pc_lat = pc(lon, lat)
    pc_lon = np.array(pc_lon)
    pc_lat = np.array(pc_lat)
    # Calculate delta_lon and delta_lat on the projected grid
    delta_lon = abs(np.diff(pc_lon))
    non_zero_delta_lon = delta_lon[delta_lon != 0]
    delta_lat = abs(np.diff(pc_lat))
    non_zero_delta_lat = delta_lat[delta_lat != 0]

    # Define a regular grid for interpolation
    n_pix_lon = int(np.floor(abs(pc_lon.max() - pc_lon.min()) / abs(np.min(non_zero_delta_lon))))  # around 400 for O96
    n_pix_lat = int(np.floor(abs(pc_lat.max() - pc_lat.min()) / abs(np.min(non_zero_delta_lat))))  # around 192 for O96
    regular_pc_lon = np.linspace(pc_lon.min(), pc_lon.max(), n_pix_lon)
    regular_pc_lat = np.linspace(pc_lat.min(), pc_lat.max(), n_pix_lat)
    grid_pc_lon, grid_pc_lat = np.meshgrid(regular_pc_lon, regular_pc_lat)

    yt = y_true.squeeze()
    yp = y_pred.squeeze()

    # check for any nan in yt
    nan_flag = np.isnan(yt).any()

    method = "linear" if nan_flag else "cubic"
    yt_i = griddata((pc_lon, pc_lat), yt, (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)
    yp_i = griddata((pc_lon, pc_lat), yp, (grid_pc_lon, grid_pc_lat), method=method, fill_value=0.0)

    # Masking NaN values
    if nan_flag:
        mask = np.isnan(yt_i)
        if mask.any():
            yt_i = np.where(mask, 0.0, yt_i)
            yp_i = np.where(mask, 0.0, yp_i)

    amplitude_t = np.array(compute_spectra(yt_i))
    amplitude_p = np.array(compute_spectra(yp_i))

    return amplitude_t, amplitude_p


# =============================================================================
# Metrics temporal average
# =============================================================================
def bias(frcst: xr.DataArray, target: xr.DataArray):
    """Calculate the bias between two DataArrays at each location.

    Args:
    frcst: xarray.DataArray
        Forecast DataArray.
    target: xarray.DataArray
        Target DataArray.
    
    Returns:
    xarray.DataArray
        Root mean square error between the forecast and target DataArrays.
    """
    dims=('init_time')
    return (frcst - target).mean(dim=dims) 

def rmse(frcst: xr.DataArray, target: xr.DataArray):
    """Calculate the area weighted root mean square error between two DataArrays.

    Args:
    frcst: xarray.DataArray
        Forecast DataArray.
    target: xarray.DataArray
        Target DataArray.
    
    Returns:
    xarray.DataArray
        Root mean square error between the forecast and target DataArrays.
    """
    dims=('init_time')
    return np.sqrt(((frcst - target)**2).mean(dim=dims)) 

def correlation(frcst: xr.DataArray, target: xr.DataArray):
    """Calculate the spatial correlation between two DataArrays.

    Args:
    frcst: xarray.DataArray
        Forecast DataArray.
    target: xarray.DataArray
        Target DataArray.
    
    Returns:
    xarray.DataArray
        Spatial correlation between the forecast and target DataArrays.
    """
    dims=('init_time')
    # Spatial anomalies
    corr = xr.corr(target, frcst, dim=dims)
    return corr