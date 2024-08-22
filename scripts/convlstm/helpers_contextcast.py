'''Helper functions for SwinLSTM model. 

@Author  :   Jakob Schlör 
@Time    :   2023/08/23 15:15:35
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import torch
import numpy as np
import xarray as xr

from forenso.utils import enso, metric


def hindcast(val_dataloader, model, device, history, horizon):
    """Hindcast Swin-LSTM model"""
    target, frcst_mean, frcst_std, time_ids = [], [], [], []
    with torch.no_grad():
        for i, (sample, aux) in enumerate(val_dataloader):
            x_input, x_target = sample.to(device).split([history, horizon], dim = 2)
            c = aux['month'].to(device = device, dtype = torch.long).argmax(dim = -1)
            x_pred = model(x_input, context=c)

            frcst_mean.append(x_pred.mean(dim=1).cpu())
            frcst_std.append(x_pred.std(dim=1).cpu())

            target.append(x_target.cpu())
            time_ids.append(aux['idx'][:, history:])

    time_idx = torch.cat(time_ids, dim=0).to(dtype=int).numpy()
    x_hindcast = {}
    x_hindcast['target'] = torch.cat(target, dim=0).numpy()
    x_hindcast['frcst_mean'] = torch.cat(frcst_mean, dim=0).numpy()
    x_hindcast['frcst_std'] = torch.cat(frcst_std, dim=0).numpy()

    return x_hindcast, time_idx


def evaluation_metric(x_hindcast: dict, time_idx: np.ndarray, 
                      val_ds: xr.DataArray, land_area_mask: xr.DataArray,
                      lag_arr: np.ndarray):

    verification_per_gridpoint, verification_per_time, nino_indices = [], [], []
    for lag in lag_arr:
        print(f"Lag: {lag}")
        ds_hindcast = dict()
        for key, x in x_hindcast.items():
            x_lst  = []
            for i, var in enumerate(val_ds.data_vars):
                da = xr.DataArray(
                    data= x[:, i, lag - 1],
                    coords=val_ds.isel(time=time_idx[:, lag - 1]).coords,
                    name = var)

                # Unnormalize
                if 'normalizer' in val_ds[var].attrs.keys():
                    da = val_ds[var].attrs['normalizer'].inverse_transform(da)
                
                # Mask land
                da = da.where(land_area_mask==0, other=np.nan)

                # Add to list
                x_lst.append(da)

            ds_hindcast[key] = xr.merge(x_lst)


        # Compute metrics
        x_target = ds_hindcast['target']
        x_frcst_mean = ds_hindcast['frcst_mean']
        x_frcst_std = ds_hindcast['frcst_std'] 

        print("Compute metrics!", flush=True)
        grid_verif = metric.verification_metrics_per_gridpoint(x_target, x_frcst_mean,
                                                               x_frcst_std)
        grid_verif['lag'] = lag
        verification_per_gridpoint.append(grid_verif)

        time_verif = metric.verification_metrics_per_time(x_target, x_frcst_mean,
                                                               x_frcst_std)
        time_verif['lag'] = lag
        verification_per_time.append(time_verif)

        # Nino indices
        nino_index = {
            'target': enso.get_nino_indices(x_target['ssta'], antimeridian=True),
            'frcst_mean': enso.get_nino_indices(x_frcst_mean['ssta'], antimeridian=True),
            'frcst_std':enso.get_nino_indices(x_frcst_std['ssta'], antimeridian=True),
            'lag': lag
        }
        nino_indices.append(nino_index)

    return verification_per_gridpoint, verification_per_time, nino_indices
