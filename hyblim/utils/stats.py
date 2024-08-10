import numpy as np
import pandas as pd 
import xarray as xr
import scipy.stats as stats

def ttest_field(X: xr.DataArray, Y: xr.DataArray, dim: str='time', alternative: str='two-sided'):
    """Point-wise t-test between means of samples from two distributions to test against
    the null-hypothesis that their means are equal.

    Args:
        X (xr.Dataarray): Samples of 1st distribution.
        Y (xr.Dataarray): Samples of 2nd distribution to test against.
        dim (str, optional): Dimension along which to compute the t-test. Defaults to 'time'.
        alternative (str, optional): Alternative hypothesis. Defaults to 'two-sided'.
        
    Returns:
        statistics (xr.Dataarray): T-statistics
        pvalues (xr.Dataarray): P-values
    """
    mean_x = X.mean(dim=dim, skipna=True).stack(z=('lat', 'lon'))
    std_x = X.std(dim=dim, skipna=True).stack(z=('lat', 'lon'))

    mean_y = Y.mean(dim=dim, skipna=True).stack(z=('lat', 'lon'))
    std_y = Y.std(dim=dim, skipna=True).stack(z=('lat', 'lon'))

    nobs_x = len(X[dim])
    nobs_y = len(Y[dim])

    statistic, pvalues = stats.ttest_ind_from_stats(mean_x.data, std_x.data, nobs_x,
                                                    mean_y.data, std_y.data, nobs_y,
                                                    equal_var=False,
                                                    alternative=alternative)
    # Convert to xarray
    statistic = xr.DataArray(data=statistic, coords=mean_x.coords)
    pvalues = xr.DataArray(data=pvalues, coords=mean_x.coords)
    return statistic.unstack(), pvalues.unstack()


def holm(pvals, alpha=0.05, corr_type="dunn"):
    """
    Returns indices of p-values using Holm's method for multiple testing.
    """
    n = len(pvals)
    sortidx = np.argsort(pvals)
    p_ = pvals[sortidx]
    j = np.arange(1, n + 1)
    if corr_type == "bonf":
        corr_factor = alpha / (n - j + 1)
    elif corr_type == "dunn":
        corr_factor = 1. - (1. - alpha) ** (1. / (n - j + 1))
    try:
        idx = np.where(p_ <= corr_factor)[0][-1]
        idx = sortidx[:idx]
    except IndexError:
        idx = []
    return idx


def field_significance_mask(pvalues, stackdim=('lat', 'lon'), alpha=0.05, corr_type="dunn"):
    """Create mask field with 1, np.NaNs for significant values
    using a multiple test correction.

    Args:
        pvalues (xr.Dataarray): Pvalues
        alpha (float, optional): Alpha value. Defaults to 0.05.
        corr_type (str, optional): Multiple test correction type. Defaults to "dunn".

    Returns:
        mask (xr.Dataarray): Mask
    """
    if corr_type is not None:
        pvals_flat = pvalues.stack(z=stackdim)
        mask_flat = xr.DataArray(data=np.zeros(len(pvals_flat), dtype=bool),
                                 coords=pvals_flat.coords)
        ids = holm(pvals_flat.data, alpha=alpha, corr_type=corr_type)
        mask_flat[ids] = True
        mask = mask_flat.unstack()
    else:
        mask = xr.where(pvalues <= alpha, True, False)

    return mask