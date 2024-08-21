''' Perfrom hindcast of LSTM model and compute verification metrics. 

@Author  :   Jakob Schlör 
@Time    :   2023/08/23 14:40:23
@Contact :   jakob.schloer@uni-tuebingen.de
'''
# %%
import os, argparse, json
import torch
import numpy as np
import xarray as xr
from tqdm import tqdm
from hyblim.model import lstm
from hyblim.data import eof, dataloader, preproc
from hyblim.utils import metric, enso

from importlib import reload
reload(metric)

PATH = os.path.dirname(os.path.abspath(__file__))


def hindcast(model, dataloader, normalizer_pca, device, history, horizon):
    targets, frcsts, time_ids = [], [], []
    with torch.no_grad():
        unused = dataloader.dataset.n_timesteps - history - horizon  
        for sample, aux in tqdm(dataloader):
            x_input, x_target, _ = sample.to(device).split([history, horizon, unused], dim=1)
            context, _ = aux['month'].to(device=device, dtype=torch.long).split([history + horizon, unused], dim=1)
            x_ensemble = model(x_input, context)

            targets.append(x_target.cpu())
            time_ids.append(aux['idx'][:, history:])
            frcsts.append(x_ensemble.cpu())

    time_idx = torch.cat(time_ids, dim=0).to(dtype=int).numpy()
    z_hindcast = {}
    z_hindcast['target'] = torch.cat(targets, dim=0).unsqueeze(dim=1).numpy()
    z_hindcast['frcst'] = torch.cat(frcsts, dim=0).numpy()
    
    # Unnormalize in PCA space
    for key, z_norm in z_hindcast.items():
        n_time, n_member, n_lag, n_components = z_norm.shape
        # Convert to xarray with eof dimension to unnormalize each eof seperately
        z_norm = xr.DataArray(z_norm, coords=dict(
            n_time=np.arange(n_time), n_member=np.arange(n_member),
            lag=np.arange(1, n_lag+1), eof=np.arange(1, n_components+1)
        ))
        z = np.zeros((n_time, n_member, n_lag, n_components))
        for i in range(n_member):
            for j in range(n_lag):
                z[:,i, j, :] = normalizer_pca.inverse_transform(z_norm[:, i, j, :])
        z_hindcast[key] = z
    
    z_hindcast['target'] = z_hindcast['target'].squeeze()
    return z_hindcast, time_idx


def frcst_to_grid(z_hindcast: np.ndarray, times: np.ndarray, combined_eof, normalizers,
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


def evaluation_metric(z_hindcast: np.ndarray,
                      time_idx: np.ndarray,
                      times: np.ndarray,
                      combined_eof: eof.CombinedEOF,
                      ds_target: xr.Dataset,
                      lag_arr: list=[1, 3, 6, 9, 12, 15, 18, 24],
                      extended_eof: eof.CombinedEOF = None):
    """Compute verification metrics for LSTM in PCA space
    
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
        print(f"Lag: {lag}")
        x_hindcast = frcst_to_grid(
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

        print("Compute metrics!")
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

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', '--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('-data', '--datasplit', type=str, default='test', help='Datasplit to evaluate')
    parser.add_argument('-lags', '--lags', nargs='+', default=[1, 3, 6, 9, 12, 15, 18, 21, 24],
                        help='Lags to compute metrics for.')
    params = vars(parser.parse_args())
    return params

def main():
    """Main function to perform hindcast and compute verification metrics.
    
    e.g.
    params = {'model_path': f'{PATH}/../../models/lstm/589593_CSLSTM_eof_[20, 10]_g0.65-crps_member16_nhist_12_nhoriz_20_layers_2_latent64_cosinelr0.005-5e-06_bs64',
              'datasplit': 'val',
              'lags': [1, 12]}
    """
    params = argument_parser()
    with open(params['model_path'] + "/config.json", 'r') as f:
        config = json.load(f)

    # Load data
    ds, datasets, dataloaders, combined_eof, scaler_pca = dataloader.load_pcdata(**config) 

    # Define and load model
    condition_dim = 12 if config['film'] else None
    model = lstm.EncDecLSTM(input_dim=combined_eof.n_components, hidden_dim=config['hidden_dim'],
                            num_layers=config['layers'], num_conditions=condition_dim,
                            num_tails=config['members'], T_max=config['chrono'])

    # Load model with best loss
    checkpoint = torch.load(params['model_path'] + "/min_checkpoint.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _ = model.to(device)

    # Hindcast in latent space
    z_hindcast, time_idx = hindcast(model, dataloaders[params['datasplit']], scaler_pca, device,
                                    history=4, horizon=24)

    # Extended PCA with 300 components 
    n_components_full = 300 
    eofa_list = []
    for i, var in enumerate(ds.data_vars):
        print(f"Create extended EOF of {var}!", flush=True)
        eofa = eof.EmpiricalOrthogonalFunctionAnalysis(n_components=n_components_full, )
        eofa.fit(ds[var])
        eofa_list.append(eofa)
    extended_eof = eof.CombinedEOF(eofa_list, vars=list(ds.data_vars))

    # Verification metrics
    times = datasets[params['datasplit']].dataarray['time'].data
    ds_target = ds.sel(time=times)
    lag_arr = [int(lag) for lag in params['lags']]
    verification_per_gridpoint, verification_per_time, nino_indices = evaluation_metric(
        z_hindcast['frcst'], time_idx, times, combined_eof, ds_target, lag_arr, extended_eof  
    )

    # Save metrics to file
    print("Save metrics to file!", flush=True)
    storepath = params['model_path'] + "/metrics"
    if not os.path.exists(storepath):
        os.makedirs(storepath)

    for key, score in verification_per_gridpoint.items():
        score.to_netcdf(storepath + f"/gridscore_{key}_{params['datasplit']}.nc")
    for key, score in verification_per_time.items():
        score.to_netcdf(storepath + f"/timescore_{key}_{params['datasplit']}.nc")
    for key, nino_idx in nino_indices.items():
        nino_idx.to_netcdf(storepath + f"/nino_{key}_{params['datasplit']}.nc")

if __name__ == '__main__':
    main()