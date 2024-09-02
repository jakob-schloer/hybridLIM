'''Helper functions for SwinLSTM model. 

@Author  :   Jakob Schlör 
@Time    :   2023/08/23 15:15:35
@Contact :   jakob.schloer@uni-tuebingen.de
'''
# %%
import os, glob, argparse, json
import numpy as np
import xarray as xr
from tqdm import tqdm
import torch
from hyblim.model import convlstm
from hyblim.data import dataloader, preproc
from hyblim.utils import enso, metric

PATH = os.path.dirname(os.path.abspath(__file__))
id = int(os.environ.get('SLURM_LOCALID', 0))
device= torch.device("cuda", id ) if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device: {device}', flush=True)


def hindcast_evaluation(loader, model, device, history=4, horizon=24 ):
    """ Perform hindcast of ConvLSTM model and compute verification metrics.

    Args:
        loader (torch.utils.data.DataLoader): Dataloader for evaluation
        model (torch.nn.Module): Trained ConvLSTM model
        device (torch.device): Device to run model on
        history (int): Number of lagged time steps
        horizon (int): Number of forecast time steps
    
    Returns:
        verification_per_gridpoint (dict): Verification metrics per gridpoint
        verification_per_time (dict): Verification metrics per time.
        nino_indices (dict): Nino indices for forecast and target
    """
    ds = loader.dataset.data
    normalizer = {
        var: preproc.normalizer_from_dict(ds[var].attrs) for var in ds.data_vars
    }

    nino_pred, nino_target = [], []
    metrics_map, metrics_time = [], [] 
    with torch.no_grad():
        for i, (sample, aux) in tqdm(enumerate(loader)):
            n_batch, n_vars, n_time, n_lat, n_lon = sample.shape 
            unused = n_time - history - horizon
            x_input, x_target, _ = sample.to(device).split([history, horizon, unused], dim=2)
            context, _ = aux['month'].to(device, dtype=torch.long).split(
                [history + horizon, unused], dim=-1
            )
            # (batch, member, vars, lag, lat, lon)
            x_pred = model(x_input, context=context)

            # Convert to xarray
            dates_batch = ds.time[aux['idx'][:, history].numpy().astype(int)]
            dims = ['time', 'member', 'lag', 'lat', 'lon']
            coords = {'time': dates_batch, 
                      'member': np.arange(1, config['member'] + 1),
                      'lag': np.arange(1, horizon + 1),
                      'lat': ds.lat, 'lon': ds.lon}
            xr_pred = metric.torch_to_xarray(x_pred.permute(2, 0, 1, 3, 4, 5), list(ds.data_vars), dims, **coords)
            # Unnormalize
            xr_pred = xr.merge(
                [normalizer[var].inverse_transform(xr_pred[var]) for var in xr_pred.data_vars]
            )

            dims = ['time', 'lag', 'lat', 'lon']
            coords = {'time': dates_batch, 
                      'lag': np.arange(1, horizon + 1),
                      'lat': ds.lat, 'lon': ds.lon}
            xr_target = metric.torch_to_xarray(x_target.permute(1, 0, 2, 3, 4), list(ds.data_vars), dims, **coords)
            # Unnormalize
            xr_target = xr.merge(
                [normalizer[var].inverse_transform(xr_target[var]) for var in xr_target.data_vars]
            )

            # Metrics
            nino_target.append(enso.get_nino_indices(xr_target['ssta']))
            nino_pred.append(enso.get_nino_indices(xr_pred['ssta']))
            metrics_map.append(
                metric.verification_metrics_per_gridpoint(
                    xr_target, xr_pred.mean(dim='member'), xr_pred.std(dim='member'), xr_pred.dims['member']
                )
            )
            metrics_time.append(
                metric.verification_metrics_per_time(
                    xr_target, xr_pred.mean(dim='member'), xr_pred.std(dim='member'), xr_pred.dims['member']
                )
            )

    # Nino indices 
    nino_indices = {
        'frcst': xr.concat(nino_pred, dim='time').sortby('time'),
        'target': xr.concat(nino_target, dim='time').sortby('time')
    }

    # Compute metrics per time
    metrics_time = metric.listofdicts_to_dictoflists(metrics_time)
    for key, score in metrics_time.items():
        metrics_time[key] = xr.concat(score, dim='time').sortby('time')

    # Metics per gridpoint
    metrics_map = metric.listofdicts_to_dictoflists(metrics_map)
    for key, score in metrics_map.items():
        metrics_map[key] = xr.concat(score, dim='time').mean('time')
    
    return metrics_map, metrics_time, nino_indices


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', '--model_path', type=str, required=True, help='Path to model')
    parser.add_argument('-datasplit', '--datasplit', type=str, default='test', help='Datasplit to evaluate')
    params = vars(parser.parse_args())
    return params

# %%
if __name__ == '__main__':
    # Specify parameters
    params = argument_parser()

# %%
params = {
    'model_path': "/home/ecm1922/Code/hybridLIM/models/convlstm/30458451_ConvLSTM_ssta_ssha_g0.65-crps_member16_nhist_12_nhoriz_20_layers_2_ch256_cosinelr0.008-1e-06_bs32",
    'datasplit': 'test',
}
with open(params['model_path'] + "/config.json", 'r') as f:
    config = json.load(f)

config['batch_size'] = 16
# Load data
ds, datasets, dataloaders = dataloader.load_stdata(**config)
config['input_dim'] = len(list(ds.data_vars))

# Define and load model
condition_num = 12 if config['film'] else -1
model = convlstm.EncDecSwinLSTM(
         input_dim=config['input_dim'],
         num_channels=config['num_channels'],
         output_dim=config['input_dim'],
         patch_size=(4,4),
         num_layers= config['num_layers'],
         num_conditions=condition_num,
         num_tails=config['members']
).to(device) 

# Load model with best loss
ckpt_file = glob.glob(params['model_path'] + "/best-checkpoint-epoch*")[0]
checkpoint = torch.load(ckpt_file, map_location=device)


# %%
# Check forescast example
loader = dataloaders[params['datasplit']]
history, horizon = 4, 24


ds = loader.dataset.data
normalizer = {
    var: preproc.normalizer_from_dict(ds[var].attrs) for var in ds.data_vars
}

nino_pred, nino_target = [], []
metrics_map, metrics_time = [], [] 
with torch.no_grad():
    for i, (sample, aux) in tqdm(enumerate(loader)):
        n_batch, n_vars, n_time, n_lat, n_lon = sample.shape 
        unused = n_time - history - horizon
        x_input, x_target, _ = sample.to(device).split([history, horizon, unused], dim=2)
        context, _ = aux['month'].to(device, dtype=torch.long).split(
            [history + horizon, unused], dim=-1
        )
        # (batch, member, vars, lag, lat, lon)
        x_pred = model(x_input, context=context)

        # Convert to xarray
        dates_batch = ds.time[aux['idx'][:, history].numpy().astype(int)]
        dims = ['time', 'member', 'lag', 'lat', 'lon']
        coords = {'time': dates_batch, 
                  'member': np.arange(1, config['members'] + 1),
                  'lag': np.arange(1, horizon + 1),
                  'lat': ds.lat, 'lon': ds.lon}
        xr_pred = metric.torch_to_xarray(x_pred.permute(2, 0, 1, 3, 4, 5), list(ds.data_vars), dims, **coords)
        # Unnormalize
        xr_pred = xr.merge(
            [normalizer[var].inverse_transform(xr_pred[var]) for var in xr_pred.data_vars]
        )

        dims = ['time', 'lag', 'lat', 'lon']
        coords = {'time': dates_batch, 
                  'lag': np.arange(1, horizon + 1),
                  'lat': ds.lat, 'lon': ds.lon}
        xr_target = metric.torch_to_xarray(x_target.permute(1, 0, 2, 3, 4), list(ds.data_vars), dims, **coords)
        # Unnormalize
        xr_target = xr.merge(
            [normalizer[var].inverse_transform(xr_target[var]) for var in xr_target.data_vars]
        )

        # Metrics
        nino_target = enso.get_nino_indices(xr_target['ssta'])
        nino_pred = enso.get_nino_indices(xr_pred['ssta'])

        break
# %%
xr_pred['ssta'].isel(lag=0, time=0).mean('member').plot()

# %%
verification_per_gridpoint, verification_per_time, nino_indices = hindcast_evaluation(
    dataloaders[params['datasplit']], model, device, history=4, horizon=24
)

scorepath = params['model_path'] + "/metrics"
# Save metrics to file
print("Save metrics to file!", flush=True)
if not os.path.exists(scorepath):
    os.makedirs(scorepath)

for key, score in verification_per_gridpoint.items():
    score.to_netcdf(scorepath + f"/gridscore_{key}_{params['datasplit']}.nc")
for key, score in verification_per_time.items():
    score.to_netcdf(scorepath + f"/timescore_{key}_{params['datasplit']}.nc")
for key, nino_idx in nino_indices.items():
    nino_idx.to_netcdf(scorepath + f"/nino_{key}_{params['datasplit']}.nc")

# %%
