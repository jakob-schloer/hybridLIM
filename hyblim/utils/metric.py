''' Collection of metrics.

@Author  :   Jakob Schlör 
@Time    :   2022/10/18 15:29:31
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import numpy as np
import pandas as pd
import xarray as xr
import torch
import scipy.stats as stats
from scipy.special import erf
from scipy.fft import fft, fftfreq

from hyblim.data import preproc
from hyblim.utils import enso

def power_spectrum(data):
    """Compute power spectrum.

    Args:
        data (np.ndarray): Data of dimension (n_feat, n_time) 

    Returns:
        xf (np.ndarray): Frequencies of dimension (n_time//2) 
        yf (np.ndarray): Power spectrum of dimension (n_feat, n_time//2) 
    """
    n_feat, n_time = data.shape
    yf = []
    for i in range(n_feat):
        yf.append(fft(data[i,:]))

    xf = fftfreq(n_time, 1)[:n_time//2]
    yf = 2./n_time * np.abs(yf)[:, :n_time//2]
    return xf, yf 


def correlation_coefficient(x: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """Correlation coefficient.

    Args:
        x (np.ndarray): Ground truth data of size (n_samples, n_features)
        x_hat (np.ndarray): Prediction of size (n_samples, n_features)

    Returns:
        cc (np.ndarray): CC of size (n_features) 
    """
    cc = np.array(
        [stats.pearsonr(x[:, i], x_hat[:, i])[0]
         for i in range(x.shape[1])]
    )
    return cc


def crps_gaussian(x: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """CRPS for Gaussian distribution.

    Args:
        x (np.ndarray): Ground truth data of size (n_samples, n_features)
        mu (np.ndarray): Mean of size (n_samples, n_features)
        std (np.ndarray): Standard deviation of size (n_samples, n_features)

    Returns:
        crps (np.ndarray): CRPS of size (n_samples, n_features) 
    """
    sqrtPi = np.sqrt(np.pi)
    z = (x - mu) / std 
    phi = np.exp(-z ** 2 / 2) / (np.sqrt(2) * sqrtPi) #standard normal pdf
    crps = std * (z * erf(z / np.sqrt(2)) + 2 * phi - 1 / sqrtPi) #crps as per Gneiting et al 2005
    return crps 


def crps_empirical(x_target: np.ndarray, x_pred: np.ndarray):
    """CPRS for empirical distribution.

    Args:
        x_target (np.ndarray): Target data of shape (num_samples, num_feature1, ...)
        x_pred (np.ndarray): Predicted data of shape (num_samples, num_feature1, ...)

    Returns:
        cprs_nrg (np.ndarray): CRPS for each feature (num_feature1, ...)
    """
    num_samples = x_pred.shape[0]
    absolute_error = np.mean(np.abs(x_pred - x_target), axis=0)

    x_pred = np.sort(x_pred, axis=0)
    diff = x_pred[1:] - x_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)

    # Expand dimensions of weight
    ndim = [1] * x_pred.ndim
    ndim[0] = len(weight)
    weight = weight.reshape(ndim)

    crps_nrg = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    return crps_nrg


def verification_metrics_per_gridpoint(target: xr.Dataset, 
                                       frcst_mean: xr.Dataset,
                                       frcst_std: xr.Dataset=None,
                                       n_members: int=16) -> dict:
    """Verification metrics for forecast in data space for each time step seperately.

    Args:
        target (xr.Dataset): Target.
        frcst_mean (xr.Dataset): Forecast mean. 
        frcst_std (xr.Dataset, optional): Forecast std.. Defaults to None.
        n_member (int, optional): Number of ensemble members. Defaults to 16.

    Returns:
        dict: Dictionary of metrics.
    """
    verification_metrics = {}
    # Point metrics
    # MSE
    mse = ((target - frcst_mean)**2).mean(dim='time', skipna=True)
    verification_metrics['mse'] = mse

    # RMSE skill score
    skill = 1 - np.sqrt(mse) / target.std(dim='time', skipna=True, ddof=1)
    verification_metrics['rmsess'] = skill

    # Correlation coefficient
    verification_metrics['cc'] = xr.merge([
            xr.corr(target[var], frcst_mean[var], dim='time') 
            for var in target.data_vars
    ])

    # Ensemble metrics
    if frcst_std is not None:
        # CRPS
        crps = crps_gaussian(target, frcst_mean, frcst_std)
        verification_metrics['crps'] = crps.mean(dim='time', skipna=True)
        verification_metrics['crpss'] = crps.mean(dim='time', skipna=True) / target.std(dim='time', skipna=True, ddof=1)

        # Spread to skill ratio 
        spread = np.square(frcst_std).mean(dim='time', skipna=True)
        spread_skill = np.sqrt( (n_members + 1)/ n_members ) * spread / mse 
        verification_metrics['spread'] = spread
        verification_metrics['spread_skill'] = spread_skill


    return verification_metrics


def verification_metrics_per_time(target: xr.Dataset, 
                                  frcst_mean: xr.Dataset,
                                  frcst_std: xr.Dataset=None,
                                  n_members: int=16) -> dict:
    """Verification metrics for forecast in data space for each time step seperately.

    Args:
        target (xr.Dataset): Target.
        frcst_mean (xr.Dataset): Forecast mean. 
        frcst_std (xr.Dataset, optional): Forecast std.. Defaults to None.
        n_member (int, optional): Number of ensemble members. Defaults to 16.

    Returns:
        dict: Dictionary of metrics.
    """
    verification_metrics = {}
    # Time metrics
    # MSE
    mse = ((target - frcst_mean)**2).mean(dim=('lat', 'lon'), skipna=True)
    verification_metrics['mse'] = mse

    # RMSE skill score
    skill = 1 - np.sqrt(mse) / target.std(dim=('lat', 'lon'), skipna=True, ddof=1)
    verification_metrics['rmsess'] = skill

    # Correlation coefficient
    cc = xr.merge([
            xr.corr(target[var], frcst_mean[var], dim=('lat', 'lon')) 
            for var in target.data_vars
    ])
    verification_metrics['cc'] = cc

    # Ensemble metrics
    if frcst_std is not None:
        # CRPS
        crps = crps_gaussian(target, frcst_mean, frcst_std)
        verification_metrics['crps'] = crps.mean(dim=('lat', 'lon'), skipna=True)
        verification_metrics['crpss'] = 1 - crps.mean(dim=('lat', 'lon'), skipna=True) / target.std(dim=('lat', 'lon'), skipna=True, ddof=1)

        # Spread to skill ratio 
        spread = np.square(frcst_std).mean(dim=('lat', 'lon'), skipna=True)
        spread_skill = np.sqrt( (n_members + 1)/ n_members ) * spread / mse 
        verification_metrics['spread'] = spread
        verification_metrics['spread_skill'] = spread_skill

    return verification_metrics


def time_series_score(frcst: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    """Compute time series scores.

    Args:
        frcst (xr.Dataset): Forecast of dimension (time, lat, lon).
        target (xr.Dataset): Target data of dimension (time, lat, lon).
    Returns:
        scores (dict): Dictionary of scores. 
        scores_vmonth (dict): Dictionary of scores for each verification month.
    """
    if 'member' in frcst.dims:
        frcst_mean = frcst.mean(dim='member')
        frcst_std = frcst.std(dim='member') 
        n_members = frcst.dims['member']
    else:
        frcst_mean = frcst 
        frcst_std = None

    # Metrics averaged over the whole time series 
    scores = dict() 
    # MSE
    mse = ((target - frcst_mean)**2).mean(dim='time', skipna=True)
    scores['mse'] = mse
    # RMSE skill score
    rmse = np.sqrt(mse)
    scores['rmse'] = rmse
    std = target.std(dim='time', skipna=True)
    scores['rmsess'] = 1 - rmse/std
    # Correlation coefficient
    scores['cc'] = xr.merge([
            xr.corr(target[var], frcst_mean[var], dim='time') for var in target.data_vars
    ])
    if frcst_std is not None:
        # CRPS
        crps = crps_gaussian(target, frcst_mean, frcst_std)
        crps_reference = crps_gaussian(target, xr.zeros_like(frcst_mean), std*xr.ones_like(frcst_std))
        scores['crps'] = crps.mean(dim='time')
        scores['crpss'] = 1 - (crps.mean(dim='time')/ crps_reference.mean(dim='time'))

        # Spread to skill ratio 
        spread = np.square(frcst_std).mean(dim='time', skipna=True)
        scores['spread'] = spread
        scores['spread_skill'] = np.sqrt( (n_members + 1)/ n_members ) * spread / mse 

    # Metrics for each verification month 
    # ===========================================
    scores_vmonth = dict()
    std_month = target.groupby(f'time.month').std(dim=('time'), skipna=True)

    # MSE
    mse_month = ((target - frcst_mean)**2).groupby(f'time.month').mean(dim=('time'), skipna=True)
    scores_vmonth['mse'] = mse_month
    scores_vmonth['rmse'] = np.sqrt(mse_month)
    # RMSE skill score
    rmse_month = np.sqrt(mse_month)
    scores_vmonth['rmsess'] = 1 - rmse_month/std_month

    if frcst_std is not None:
        # CRPS
        crps_month = crps_gaussian(target, frcst_mean, frcst_std).groupby(f'time.month').mean(dim=('time'))
        scores_vmonth['crps'] = crps_month
        scores_vmonth['crpsss'] = crps_month / std_month

        # Spread to skill ratio 
        spread_month = np.square(frcst_std).groupby('time.month').mean(dim='time')
        scores_vmonth['spread'] = spread_month
        scores_vmonth['spread_skill'] = np.sqrt( (n_members + 1)/ n_members ) * spread_month / mse_month 

    return scores, scores_vmonth 


def frcst_metrics_per_month(target: xr.Dataset, frcst: xr.Dataset) -> dict:
    """Metrics for forecast in data space for each month seperately.

    Args:
        target (xr.Dataset): Target data of dimension (time, lat, lon).
        frcst (xr.Dataset): Forecast of dimension (time, lat, lon).

    Returns:
        dict: Metrics include: 
            - 'acc' of shape (lat, lon, month),
            - 'pattern_corr' of shape (month)
            - 'mse' of shape (lat, lon, month)
        
    """
    # Correlation metrics
    pattern_corr = []
    acc = []
    for var in list(target.data_vars):
        temp_x, idx_nNaN = preproc.map2flatten(target[var])
        temp_x_frcst, _ = preproc.map2flatten(frcst[var])

        # ACC
        print("Compute monthly ACC!")
        acc_monthly_var = []
        for m in np.unique(temp_x.time.dt.month):
            temp_acc = correlation_coefficient(
                temp_x.isel(time=np.where(temp_x.time.dt.month == m)[0]),
                temp_x_frcst.isel(time=np.where(temp_x_frcst.time.dt.month == m)[0])
            )
            acc_monthly_var.append(preproc.flattened2map(temp_acc, idx_nNaN))
        acc.append(
            xr.concat(acc_monthly_var, dim=pd.Index(np.unique(temp_x.time.dt.month), name='month'))
        )

        # Pattern correlation
        print("Compute monthly Pattern Correlation!")
        pattern_corr_temp = xr.DataArray(
            data=correlation_coefficient(temp_x.data, temp_x_frcst.data),
            coords={'time': temp_x['time']}, name=var
        )
        pattern_corr.append(
            pattern_corr_temp.groupby('time.month').mean(dim='time', skipna=True)
        )

    acc = xr.merge(acc)
    pattern_corr = xr.merge(pattern_corr)

    # MSE
    print("Compute monthly MSE!")
    se = (target - frcst)**2
    mse = se.groupby('time.month').mean(dim='time', skipna=True)

    # Store metrics
    return {'acc': acc, 'pattern_corr': pattern_corr, 'mse': mse}


def random_monthly_samples(ds: xr.Dataset, n_samples: int = 200) -> xr.Dataset:
    """Randomly samples the dataset by sampling months equally.

    Args:
        ds (xr.Dataset): Dataset to sample from 
        n_samples (int, optional): Number of samples. Defaults to 200.

    Returns:
        xr.Dataset: Subsampled dataset
    """
    # Get the unique months in the dataset
    months = np.unique(ds.time.dt.month)

    # Calculate the number of samples to draw per month
    n_per_month = int(np.ceil(n_samples / len(months)))

    # Initialize an empty list to hold the selected samples
    selected_samples = []

    # Loop over the months and select samples
    for month in months:
        # Get the indices of the time steps in the current month
        month_indices = np.where(ds.time.dt.month == month)[0]

        # Draw n_per_month random indices from the current month
        selected_indices = np.random.choice(month_indices, size=min(
            n_per_month, len(month_indices)), replace=True)

        # Add the selected samples to the list
        selected_samples.append(ds.isel(time=selected_indices))

    # Concatenate the selected samples into a new dataset
    selected_ds = xr.concat(selected_samples, dim='time')

    return selected_ds


def mean_diff(model1_skill: xr.DataArray, model2_skill: xr.DataArray,
              dim: str ='sample') -> list:
    """Compute difference of means and check for their statistical significance.
    
    We use the t-test for two sample groups and one sample groups.

    Args:
        model1_skill (xr.DataArray): Skill of model 1 with dimension 'ensemble'. 
        model2_skill (xr.DataArray): Skill of model 2 with dimension 'ensemble'.
        dim (str): Name of dimension the statistics is computed over

    Returns:
        diff (xr.DataArray): Difference of ensemble means.
        pvalues (xr.DataArray): Pvalues of ttest.
    """
    diff = model1_skill.mean(dim=dim, skipna=True) - model2_skill.mean(dim=dim, skipna=True)
    axis = int(np.where(np.array(model1_skill.dims) == dim)[0])
    if (len(model1_skill[dim]) > 1) and (len(model2_skill[dim]) > 1):
        statistic, pvalue = stats.ttest_ind(model1_skill.data, model2_skill.data, axis=axis, alternative='two-sided') 
    elif (len(model1_skill[dim]) > 1) and (len(model2_skill[dim]) == 1):
        statistic, pvalue = stats.ttest_1samp(model1_skill.data, model2_skill.mean(dim=dim), axis=axis, alternative='two-sided') 
    elif (len(model1_skill[dim]) == 1) and (len(model2_skill[dim]) > 1):
        statistic, pvalue = stats.ttest_1samp(model2_skill.data, model1_skill.mean(dim=dim), axis=axis, alternative='two-sided') 
    else:
        print(f"No samples given!")
        return diff, None

    pvalues = xr.DataArray(
        pvalue, coords=diff.coords,
    )

    return diff, pvalues


def listofdicts_to_dictoflists(array_of_dicts: list) -> dict:
    """Convert list of dictionaries with same keys and xarray.datasets of same dimensions into a dictionary of xarray.

    Args:
        array_of_dicts (list): List of dictionaries with same keys. 

    Returns:
        dict: Dictionary of xr.DataSets.
    """
    dict_of_arrays = {}
    for key in array_of_dicts[0].keys():
        dict_of_arrays[key] = [d[key] for d in array_of_dicts]
    return dict_of_arrays


def listofdicts_to_dictofxr(list_of_dict, dim_key='lag'):
    dict_of_xr = {}
    dict_of_list = listofdicts_to_dictoflists(list_of_dict)
    coords = dict_of_list.pop(dim_key)
    for key, ds in dict_of_list.items():
        tmp_ds = xr.concat(ds, dim=pd.Index(coords, name=dim_key),
                            compat='equals', join='inner') 
        dict_of_xr[key] = tmp_ds 
    return dict_of_xr


def torch_to_xarray(tensor, var_names, dims, **coords):
    """ Convert a tensor to an xarray.Dataset with the first dimension as variables and the remaining 
    dimensions specified by the coords kwargs.

    Parameters:
        tensor (torch.Tensor): Input tensor of shape (batch, members, vars, ...).
        var_names (list of str): List of variable names corresponding to the first dimension.
        dims (list of str): List of dimension names for the remaining dimensions.
        **coords: Keyword arguments where each key is a dimension name and each value is a list or array of coordinate values.

    Returns:
        xr.Dataset: Dataset with each variable as a DataArray of dimensions specified by the coords kwargs.
    """
    assert tensor.shape[0] == len(var_names), "The length of var_names must match the size of the first dimension of the tensor."
    da_vars = {}
    for i, var_name in enumerate(var_names):
        da_vars[var_name] = xr.DataArray(
            tensor[i, ...].detach().cpu().numpy(),
            dims=dims,
            coords=coords 
        )
    return xr.Dataset(da_vars)
