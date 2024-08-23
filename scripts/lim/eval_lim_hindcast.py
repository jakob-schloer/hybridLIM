''' Perfrom hindcast of LSTM model and compute verification metrics. 

@Author  :   Jakob Schlör 
@Time    :   2023/08/23 14:40:23
@Contact :   jakob.schloer@uni-tuebingen.de
'''
# %%
import os, argparse
import numpy as np
import xarray as xr
from hyblim.data import eof, preproc
from hyblim.utils import metric, eval, enso

from importlib import reload
reload(metric)

PATH = os.path.dirname(os.path.abspath(__file__))


def hindcast_evaluation(lim_hindcast: xr.DataArray, 
                        ds_target: xr.Dataset, 
                        combined_eof: eof.CombinedEOF, 
                        lag_arr: list,
                        add_eof_noise: bool = True):
    """ Perform hindcast and compute verification metrics for LIM ensemble hindcasts.

    Args:
        lim_hindcast (xr.DataArray): Hindcast data.
        ds_target (xr.Dataset): Target dataset.
        combined_eof (eof.CombinedEOF): Combined EOF object
        lag_arr (list): List of lags to compute metrics for
        scorepath (str): Path to save metrics
    """
    normalizers = {var: preproc.normalizer_from_dict(ds_target[var].attrs) for var in ds_target.data_vars}

    # Extended PCA with 300 components 
    if add_eof_noise:
        n_components_full = 300 
        eofa_list = []
        for i, var in enumerate(ds_target.data_vars):
            print(f"Create extended EOF of {var}!", flush=True)
            eofa = eof.EmpiricalOrthogonalFunctionAnalysis(n_components=n_components_full, )
            eofa.fit(ds_target[var])
            eofa_list.append(eofa)
        extended_eof = eof.CombinedEOF(eofa_list, vars=list(ds_target.data_vars))
    else:
        extended_eof = None
    
    lag_arr = params['lags']
    verification_per_gridpoint, verification_per_time, nino_indices = [], [], [] 
    for lag in lag_arr:
        # Transform hindcast to grid space
        print(f"Lag: {lag}", flush=True)
        z = lim_hindcast.sel(lag=lag).data[:-lag,:,:]
        print("Z shape: ", z.shape)
        target_time = lim_hindcast['time'].data[lag:]
        print("Target time: ", target_time.shape)
        x_hindcast = eval.latent_frcst_to_grid(
            z, target_time, combined_eof, normalizers, extended_eof=extended_eof
        )

        # Get target data and revert normalization
        x_target = ds_target.sel(time=target_time)
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
            'target': enso.get_nino_indices(x_target['ssta']),
            'frcst': enso.get_nino_indices(x_hindcast['ssta']),
            'lag': lag
        }
        nino_indices.append(nino_index)

        grid_scores = metric.listofdicts_to_dictofxr(verification_per_gridpoint, dim_key='lag')
        time_scores = metric.listofdicts_to_dictofxr(verification_per_time, dim_key='lag')
        nino_ids = metric.listofdicts_to_dictofxr(nino_indices, dim_key='lag')

    return grid_scores, time_scores, nino_ids


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', '--path', type=str, required=True, help='Path to model')
    parser.add_argument('-lags', '--lags', nargs='+', default=[1, 3, 6, 9, 12, 15, 18, 21, 24],
                        help='Lags to compute metrics for.')
    params = vars(parser.parse_args())
    return params

# %%
if __name__ == '__main__':
    params = argument_parser()
    # Specify parameters
    #params = {'path': '/home/ecm1922/Code/hybridLIM/models/lim/cslim_ssta-ssha/cslim_hindcast_ssta-ssha_eof20_test.nc',
    #          'datasplit': 'test', 'lags': [1, 3]}

    # Load hindcast
    lim_hindcast = xr.open_dataset(params['path'])['z'].sel(lag=slice(1, None))

    datapaths = {}
    datapaths['ssta'] = PATH + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc"
    datapaths['ssha'] = PATH + "/../../data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssha_lat-31_33_lon130_290_gr1.0.nc"
    n_eof = [20, 10]

    # %%
    # Load data
    print("Load data!", flush=True)
    da_arr = []
    for var, path in datapaths.items():
        da = xr.open_dataset(path)[var]
        # Normalize data 
        normalizer = preproc.Normalizer()
        da = normalizer.fit_transform(da)
        # Store normalizer as an attribute in the Dataarray for the inverse transformation
        da.attrs = normalizer.to_dict()
        da_arr.append(da)

    ds = xr.merge(da_arr)

    # Apply land sea mask
    lsm = xr.open_dataset(PATH+"/../../data/land_sea_mask_common.nc")['lsm']
    ds = ds.where(lsm!=1, other=np.nan)

    # Create PCA
    eofa_lst = []
    for i, var in enumerate(ds.data_vars):
        print(f"Create EOF of {var}!")
        n_components = n_eof[i] if isinstance(n_eof, list) else n_eof 
        eofa = eof.EmpiricalOrthogonalFunctionAnalysis(n_components)
        eofa.fit(
            ds[var].isel(time=slice(None, int(0.8*len(ds['time']))))
        )
        eofa_lst.append(eofa)
    combined_eof = eof.CombinedEOF(eofa_lst, vars=list(ds.data_vars))

    # %%
    # Perform hindcast evaluation
    lag_arr = [int(lag) for lag in params['lags']]
    verification_per_gridpoint, verification_per_time, nino_indices = hindcast_evaluation(
        lim_hindcast, ds, combined_eof, lag_arr, add_eof_noise=False
    )

    # %%
    # Save metrics to file
    scorepath = os.path.join(os.path.dirname(params['path']) + "/metrics")
    print("Save metrics to file!", flush=True)
    if not os.path.exists(scorepath):
        os.makedirs(scorepath)

    years = (lim_hindcast['time.year'].min().values, lim_hindcast['time.year'].max().values)
    for key, score in verification_per_gridpoint.items():
        score.to_netcdf(scorepath + f"/gridscore_{key}_{years[0]:04d}-{years[1]:04d}.nc")
    for key, score in verification_per_time.items():
        score.to_netcdf(scorepath + f"/timescore_{key}_{years[0]:04d}-{years[1]:04d}.nc")
    for key, nino_idx in nino_indices.items():
        nino_idx.to_netcdf(scorepath + f"/nino_{key}_{years[0]:04d}-{years[1]:04d}.nc")
    # %%
