""" Functions for evaluating models."""

import numpy as np
import pandas as pd
import xarray as xr
from hyblim.data import eof, preproc
from hyblim.utils import metric, enso


def latent_frcst_to_grid(z_hindcast: np.ndarray, times: np.ndarray, combined_eof, normalizers,
                  extended_eof=None):
    """ Transform forecast in latent space to grid space.

    Args:
        z_hindcast (np.ndarray): Forecast in latent space of shape (n_time, n_member, n_features)
        times (np.ndarray): Times of forecast
        combined_eof (eof.CombinedEOF): Combined EOF object
        normalizers (dict): Dictionary of normalizers for each variable
    Returns:
        x_hindcast (xr.Dataset): Forecast in grid space of shape (n_time, n_member, n_features)
    """
    n_start = 0
    x_hindcast = []
    for i, eofa in enumerate(combined_eof.eofa_lst):
        var = combined_eof.vars[i] 
        # (n_components, n_features)
        components = eofa.pca.components_
        n_end = n_start + eofa.n_components
        # (n_time, n_member, n_features)
        z_var = z_hindcast[..., n_start:n_end]
        # (n_time, n_member, n_features)
        x_var = np.einsum('ijk,kl->ijl', z_var, components)

        # Add nans and reshape to grid space
        n_times, n_members, _ = x_var.shape
        mask_map = eofa.ids_notNaN.unstack().drop_vars(['time', 'month'])
        mask_flat = mask_map.values.flatten()
        idx_mask_flat = np.where(mask_flat)[0]
        x_flat = np.ones((n_times, n_members, *mask_flat.shape)) * np.nan
        x_flat[:, :, idx_mask_flat] = x_var

        if extended_eof is not None:
            # Inverse transform of randomly sampled remaining eofs
            for n in range(n_members):
                x_rand = eof.inverse_transform_of_random_latent(
                    extended_eof.eofa_lst[i], ignore_n_components=eofa.n_components, n_time=n_times 
                ) 
                x_flat[:, i, idx_mask_flat] += x_rand

        x_map = xr.DataArray(
            data = x_flat.reshape(n_times, n_members, *mask_map.shape),
            coords = dict(time = times, 
                          member=np.arange(1, n_members+1),
                          **mask_map.coords),
            name = var
        )
        # Invert normalization
        x_map = normalizers[var].inverse_transform(x_map)
        x_hindcast.append(x_map)

    return xr.merge(x_hindcast)


def latent_evaluation(z_hindcast: np.ndarray,
                      time_idx: np.ndarray,
                      times: np.ndarray,
                      combined_eof: eof.CombinedEOF,
                      ds_target: xr.Dataset,
                      lag_arr: list=[1, 3, 6, 9, 12, 15, 18, 24],
                      extended_eof: eof.CombinedEOF = None):
    """Compute verification metrics for hindcasts in PC space. 
    
    Args:
        z_hindcast_mean (np.ndarray): Hindcast of mean in PCA space 
            of shape (n_time, n_member, n_lag, n_components)
        time_idx (np.ndarray): Time indices of shape (n_time, n_lag)
        times (xr.DataArray): Times of validation dataset. 
        combined_eof : PCA collection object
        ds_target (xr.Dataset): Target dataset.
        lag_arr (np.ndarray, optional): Array of lags to compute metrics for.
            Defaults to [1, 3, 6, 9, 12, 15, 18, 24].
    """
    # Get normalizer of grid space data
    normalizers = {var: preproc.normalizer_from_dict(ds_target[var].attrs) for var in ds_target.data_vars}

    verification_per_gridpoint, verification_per_time, nino_indices = [], [], [] 
    for lag in lag_arr:
        # Transform hindcast to grid space
        print(f"Lag: {lag}", flush=True)
        x_hindcast = latent_frcst_to_grid(
            z_hindcast[:,:, lag-1, :],
            times[time_idx[:, lag-1]],
            combined_eof,
            normalizers,
            extended_eof=extended_eof
        )

        # Get target data
        x_target = ds_target.sel(time=times[time_idx[:, lag-1]])
        # Unnormalize in data space
        x_target = xr.merge([normalizers[var].inverse_transform(x_target[var])
                             for var in x_target.data_vars])

        print("Compute metrics!", flush=True)
        x_frcst_mean = x_hindcast.mean(dim='member')
        x_frcst_std = x_hindcast.std(dim='member', ddof=1)
        n_members = len(x_hindcast['member'])

        # Compute metrics per gridpoint
        grid_verif = metric.verification_metrics_per_gridpoint(
            x_target, x_frcst_mean, x_frcst_std, n_members=n_members
        )
        grid_verif['lag'] = lag
        verification_per_gridpoint.append(grid_verif)
        
        # Compute metrics per time
        time_verif = metric.verification_metrics_per_time(
            x_target, x_frcst_mean, x_frcst_std, n_members=n_members
        )
        time_verif['lag'] = lag
        verification_per_time.append(time_verif)

        # Nino indices
        nino_index = {
            'target': enso.get_nino_indices(x_target['ssta'], antimeridian=True),
            'frcst': enso.get_nino_indices(x_hindcast['ssta'], antimeridian=True),
            'lag': lag
        }
        nino_indices.append(nino_index)

    grid_scores = metric.listofdicts_to_dictofxr(verification_per_gridpoint, dim_key='lag')
    time_scores = metric.listofdicts_to_dictofxr(verification_per_time, dim_key='lag')
    nino_ids = metric.listofdicts_to_dictofxr(nino_indices, dim_key='lag')
    
    return grid_scores, time_scores, nino_ids 