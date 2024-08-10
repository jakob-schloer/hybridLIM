''' Collection of Dataset classes. 

@Author  :   Jakob Schlör 
@Time    :   2023/05/09 14:46:52
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import numpy as np
import xarray as xr
import torch
from torch.utils.data import DataLoader, Dataset

from hyblim.data import preproc, eof
from hyblim.model import lim

class ForConvolution(object):
    """Transformation for convolutional input layers.
    Replace NaNs by zeros."""

    def __call__(self, sample):
        for key in sample.keys():
            if key in ['data', 'target', 'input']:
                buff = np.copy(sample[key])
                # set NaNs to value
                SETNANS = 0.0
                idx_nan = np.isnan(buff)
                buff[idx_nan] = float(SETNANS)

                # change dim from (n_lon, n_lat) to (1, n_lon, n_lat)
                if len(sample[key].shape) == 2:
                    sample[key] = np.array([buff])
                else:
                    sample[key] = buff
            else:
                continue

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        torch_sample = sample
        for key in sample.keys():
            try:
                torch_sample[key] = torch.from_numpy(sample[key]).float()
            except:
                None
        
        return torch_sample


############################################
# Dataclasses 
############################################
class TimeSeries(Dataset):
    def __init__(self, dataarray, n_timesteps, transform=None):
        self.transform = transform
        self.dataarray = dataarray.compute()

        self.n_timesteps = n_timesteps


    def __len__(self):
        return len(self.dataarray['time']) - self.n_timesteps - 1


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_interval = self.dataarray.isel(time=slice(idx, idx+self.n_timesteps))
        # One hot encoding of month
        idx_month = x_interval['time'].dt.month.data.astype(int) - 1
        one_hot_month = np.zeros((len(idx_month), 12))
        one_hot_month[np.arange(len(idx_month)), idx_month] = 1

        
        sample = {
            'x':   x_interval.data.T,
            'time': preproc.time2timestamp(x_interval['time'].data),
            'month': one_hot_month,
            'idx': np.arange(idx, idx+self.n_timesteps)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample['x'], {'time': sample['time'], 'month': sample['month'], 'idx': sample['idx']}  



class TimeSeriesResidual(Dataset):
    """Dataclass for training residuals of model

    Args:
        arget (xr.DataArray): Target time-series.
            of shape (n_comp, n_times)
        lim_ensemble (xr.DataArray): LIM hindcast ensemble  
            of shape (n_comp, n_members, n_times, n_lag)
        transform (function, optional): Transform function. Defaults to None.
    """
    def __init__(self, target:xr.DataArray, lim_ensemble: xr.DataArray, transform= None):
        self.dataset = target.compute()
        self.lim_ensemble = lim_ensemble.compute() 
        self.lag_arr = lim_ensemble['lag'].values
        self.transform = transform

        assert len(self.dataset['time']) == len(self.lim_ensemble['time'])


    def __len__(self):
        return len(self.lim_ensemble['time']) - np.max(self.lag_arr)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        # Input
        sample['lim'] = self.lim_ensemble.isel(time=idx).data
        
        # Target
        ids_target = idx + self.lag_arr
        sample['target'] = self.dataset.isel(time=ids_target).data.T

        # One hot encoding of month
        idx_month_target = self.dataset['time'][ids_target].dt.month.data.astype(int) - 1
        one_hot_target = np.zeros((len(idx_month_target), 12))
        one_hot_target[np.arange(len(idx_month_target)), idx_month_target] = 1


        sample['idx'] = ids_target
        sample['month'] = one_hot_target 
        sample['time'] = preproc.time2timestamp(
                self.dataset.isel(time=ids_target)['time'].data
            )
        sample['lag'] = self.lag_arr

        if self.transform:
            sample = self.transform(sample)

        # Output
        lim_input = sample['lim']
        target = sample['target']
        label = {'time': sample['time'], 'idx': sample['idx'],
                 'month': sample['month'], 'lag': sample['lag']}

        return lim_input, target, label 


class SpatialTemporalData(Dataset):
    """Spatial temporal data of multible variables.. 

    Args:
        dataset (xr.Dataset): Dataset with dimensions ['time', 'lat', 'lon'].
        n_timesteps (int): Length of time-series. 
        transform ([type], optional): [description]. Defaults to None.
    """

    def __init__(self, dataset: xr.Dataset, n_timesteps: int,
                 transform=None):
        super().__init__()

        self.dataset = dataset.transpose('time', 'lat', 'lon').compute()
        self.vars = list(self.dataset.data_vars)
        self.coords = list(self.dataset.coords)
        self.dims = self.dataset[self.vars[0]].shape
        self.n_timesteps = n_timesteps
        
        self.transform = transform
    

    def __len__(self)-> int: 
        """Number of samples in dataset."""
        return len(self.dataset['time']) - self.n_timesteps - 1
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x_interval = self.dataset.isel(time=slice(idx, idx+self.n_timesteps))

        # One hot encoding of month
        idx_month = x_interval['time'].dt.month.data.astype(int) - 1
        one_hot_month = np.zeros((len(idx_month), 12))
        one_hot_month[np.arange(len(idx_month)), idx_month] = 1
        
        sample = {
            'data':  np.array([x_interval[var].data for var in self.vars]),
            'time':  preproc.time2timestamp(self.dataset['time'].data[idx]),
            'month': one_hot_month,
            'idx': np.arange(idx, idx+self.n_timesteps)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample['data'], {'time': sample['time'], 'month': sample['month'], 'idx': sample['idx']} 

    
    def torch2xr(self, data: torch.Tensor) -> xr.Dataset:
        """Convert torch.Tensor back to xr.Dataset.

        Args:
            data (torch.Tensor): Torch tensor with data of shape (batch, var, time, lat, lon) or (var, time, lat, lon) 
            name (str, optional): Name of xarray. Defaults to None.
            dim_name (str, optional): Name of additional dimension. Defaults to 'time'.
            dim ([type], optional): Coordinates of dimension. Defaults to None.

        Returns:
            (xr.Dataset or list of xr.Dataset): Data converted back to xr.Dataset 
        """
        if torch.is_tensor(data):
            data = data.to('cpu').detach().numpy()

        # For single sample 
        if len(data.shape) == 4:
            sample = data
            da_arr = []
            for i, var in enumerate(self.vars):
                sample_map = sample[i, :, :, :].copy()
                time_idx = np.arange(sample_map.shape[0])
                coords = {f'time_idx': time_idx,
                          'lat': self.dataset.coords['lat'],
                          'lon': self.dataset.coords['lon']}

                da_arr.append(xr.DataArray(
                    data=sample_map,
                    coords=coords,
                    name=var
                ))

            ds = xr.merge(da_arr)

        # For batch 
        elif len(data.shape) == 5:
            ds = []
            for sample in data:
                da_arr = []
                for i, var in enumerate(self.vars):
                    data_map = sample[i, :, :, :].copy()
                    time_idx = np.arange(data_map.shape[0])
                    coords = {f'time_idx': time_idx,
                              'lat': self.dataset.coords['lat'],
                              'lon': self.dataset.coords['lon']}

                    da_arr.append(xr.DataArray(
                        data=data_map,
                        coords=coords,
                        name=var
                    ))

                ds.append(xr.merge(da_arr))

        return ds


class SpatioTemporalResidual(Dataset):
    """Dataclass for training residuals of model

    Args:
        dataset_target (xr.Dataset): Target dataset.
            of shape (n_comp, n_times)
        dataset_lim (xr.DataArray): Input dataset from lim 
            of shape (n_comp, n_times, n_lag)
        hist (int): Number of history. Defaults to 0.
        transform (function, optional): Transform function. Defaults to None.
    """
    def __init__(self, dataset_target:xr.DataArray, dataset_lim: xr.DataArray, 
                 hist: int = 0, transform= None):
        self.dataset = dataset_target.transpose('time', 'lat', 'lon').compute()
        self.dataset_lim = dataset_lim.transpose('time', 'lag', 'lat', 'lon').compute()
        self.hist = hist
        self.lag_arr = dataset_lim['lag'].data
        self.vars = list(self.dataset.data_vars)
        self.coords = list(self.dataset.coords)
        self.dims = self.dataset[self.vars[0]].shape

        self.transform = transform

        assert len(self.dataset_lim['time']) == len(self.dataset['time'])

    def __len__(self):
        return len(self.dataset_lim['time']) - np.max(self.lag_arr) - self.hist 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Input
        if self.hist > 0:
            x_hist = self.dataset.isel(time=slice(idx, idx+self.hist))
            x_lim = self.dataset_lim.isel(time=idx+self.hist)
            x_input = np.array([
                np.concatenate([x_hist[var].data, x_lim[var].data], axis=0) 
                for var in x_hist.data_vars
            ])
        else:
            x_input = self.dataset_lim.isel(time=idx)
            x_input = np.array([x_input[var].data for var in self.vars])
        
        # Target
        ids_target = idx + self.hist + self.lag_arr
        x_target = self.dataset.isel(time=ids_target)

        # One hot encoding of month
        idx_month_target = self.dataset['time'][ids_target].dt.month.data.astype(int) - 1
        one_hot_target = np.zeros((len(idx_month_target), 12))
        one_hot_target[np.arange(len(idx_month_target)), idx_month_target] = 1

        sample = {
            'input': x_input,
            'target': np.array([x_target[var].data for var in self.vars]),
            'idx': ids_target,
            'month': one_hot_target, 
            'time': preproc.time2timestamp(
                self.dataset.isel(time=ids_target)['time'].data
            )
        }

        if self.transform:
            sample = self.transform(sample)

        label = {'time': sample['time'], 'idx': sample['idx'], 'month': sample['month']}

        return sample['input'], sample['target'], label 

# ======================================================================================
# Functions to load data for each class  
# ======================================================================================
def load_stdata(datapaths: dict,
    hist: int = 18, horiz: int = 24, num_traindata: int = None, 
    batch_size: int=32, **kwargs
    ) -> list:
    """Load data, cut area, interpolate on grid and split into train, val and test data.

    Args:
        datapaths (dict): Dictionaries with datasets,
            e.g. {'ts': "ts_Amon_CESM2_piControl_r1i1p1f1.nc"}.
        hist (int, optional): Number of history timesteps. Defaults to 18.
        horiz (int, optional): Number of horizon timesteps. Defaults to 24.
        num_traindata (int, optional): Number of training datapoints.
            Defaults to None, then use 75% of the data.
        batch_size (int, optional): Batch size. Defaults to 32.
        **kwargs : In case more parameters are given.
 
    Returns:
        ds (xr.Dataset): xarray datset.
        datasets (dict): Dict of torch.Datasets with keys: train, val, test
        dataloaders (dict): Dict of torch.Dataloaders with keys: train, val, test
    """
    # Load preprocessed data
    print("Load data!", flush=True)
    da_arr = []
    for var, path in datapaths.items():
        da = xr.open_dataset(path)[var]
        # Create normalizer from stored attributes
        if 'method' in da.attrs:
            normalizer = preproc.normalizer_from_dict(da.attrs)
            da.attrs['normalizer'] = normalizer
        da_arr.append(da)

    ds = xr.merge(da_arr)

    # Split in training and val data
    if num_traindata is None:
        idx_train = int(0.75*len(ds['time']))
    else:
        idx_train = num_traindata
    idx_val = int(0.75*len(ds['time']))
    idx_test = int(0.9*len(ds['time'])) 

    data = dict(
        train=ds.isel(time=slice(None,idx_train)),
        val=ds.isel(time=slice(idx_val, idx_test)),
        test=ds.isel(time=slice(idx_test, None))
    )

    # Torch dataset class
    # ======================================================================================
    n_timesteps = hist + horiz
    transform = transforms.Compose([ForConvolution(), ToTensor()])

    datasets = {}
    dataloaders = {}
    for key, values in data.items():
        datasets[key] = SpatialTemporalData(values, 
                                            n_timesteps, 
                                             transform=transform)
        shuffle = True if key == 'train' else False
        dataloaders[key] = DataLoader(datasets[key],
                                      drop_last = True,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=shuffle)

    return ds, datasets, dataloaders 


def load_pcdata(
    datapaths: dict, 
    n_eof: int=20, hist: int=3, horiz: int=24, 
    num_traindata: int=None, batch_size: int=32,
    **kwargs) -> list:
    """Load dataset for training LSTM on PCs.

    Args:
        datapaths (dict): Dictionaries with datasets,
        n_eof (int, optional): Number of EOFs. Defaults to 20.
        hist (int, optional): History length. Defaults to 3.
        horiz (int, optional): Horizon length of timeseries. Defaults to 24.
        num_traindata (int, optional): Number of training datapoints. 
            Defaults to None, then we train on 75% of the data.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        ds (xr.Dataset): xarray datset.
        datasets (dict): Dict of torch.Datasets with keys: train, val, test
        dataloaders (dict): Dict of torch.Dataloaders with keys: train, val, test
        pca_coll (eof.PCACollection): PCA collection.
        normalizer_pca (preproc.Normalizer): Normalizer for PCs.
    """

    # Load preprocessed data
    print("Load data!", flush=True)
    da_arr = []
    for var, path in datapaths.items():
        da = xr.open_dataset(path)[var]
        # Create normalizer from stored attributes
        if 'method' in da.attrs:
            normalizer = preproc.normalizer_from_dict(da.attrs)
            da.attrs['normalizer'] = normalizer
        da_arr.append(da)

    ds = xr.merge(da_arr)

    # Create PCA
    pca_lst = []
    for i, var in enumerate(ds.data_vars):
        print(f"Create EOF of {var}!")
        n_components = n_eof // (i+1)
        pca_lst.append(
            eof.SpatioTemporalPCA(ds[var], n_components=n_components),
        )
    pca_coll = eof.PCACollection(pca_lst)
    n_components = pca_coll.n_components
    pcs = pca_coll.get_principal_components()

    # Normalize
    print("Normalize PCs.", flush=True)
    normalizer_pca = preproc.Normalizer(method='zscore')
    pcs = normalizer_pca.fit_transform(pcs, dim='time')


    # Split in training and val data
    if num_traindata is None:
        idx_train = int(0.75*len(pcs['time']))
    else:
        idx_train = num_traindata

    idx_val = int(0.75*len(pcs['time']))
    idx_test = int(0.9*len(pcs['time'])) 
    data = dict(
        train=pcs.isel(time=slice(None,idx_train)),
        val=pcs.isel(time=slice(idx_val, idx_test)),
        test=pcs.isel(time=slice(idx_test, None))
    )

    # Torch dataset class
    # ======================================================================================
    n_timesteps = hist + horiz
    transform = transforms.Compose([ToTensor()])

    datasets = {}
    dataloaders = {}
    for key, values in data.items():
        datasets[key] = TimeSeries(values, 
                                   n_timesteps, 
                                   transform=transform)
        shuffle = True if key == 'train' else False
        dataloaders[key] = DataLoader(datasets[key],
                                      drop_last = True,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=shuffle)

    return ds, datasets, dataloaders, pca_coll, normalizer_pca  


def load_pcdata_lim(
    datapaths: dict, 
    n_eof: int=20, horiz: int=24, 
    num_traindata: int=None, batch_size: int=32, 
    lim_type: str='cslim', n_eof_lim: int=None, **kwargs
    ) -> list:
    """Load data for LIM+LSTM model with LIM mean prediction as input.

    Args:
        datapaths (dict): _description_
        n_eof (int, optional): _description_. Defaults to 20.
        hist (int, optional): _description_. Defaults to 0.
        horiz (int, optional): _description_. Defaults to 24.
        num_traindata (int, optional): _description_. Defaults to None.
        batch_size (int, optional): _description_. Defaults to 32.
        lim_type (str, optional): _description_. Defaults to 'cslim'.
        n_eof_lim (int, optional): _description_. Defaults to None.

    Returns:
        ds (xr.Dataset): 
        datasets (hyblim.ResidualTimeSeries):
        dataloaders (torch.Dataloader):
        pca_coll (hyblim.MultiSpatioTemporalPCA):
        normalizer_pca (hyblim.Normalizer):
    """

    # Load preprocessed data
    print("Load data!", flush=True)
    da_arr = []
    for var, path in datapaths.items():
        da = xr.open_dataset(path)[var]
        # Create normalizer from stored attributes
        if 'method' in da.attrs:
            normalizer = preproc.normalizer_from_dict(da.attrs)
            da.attrs['normalizer'] = normalizer
        da_arr.append(da)

    ds = xr.merge(da_arr)
    
    # Create PCA
    # ======================================================================================
    pca_lst = []
    for i, var in enumerate(ds.data_vars):
        print(f"Create EOF of {var}!")
        n_components = n_eof // (i+1)
        pca_lst.append(
            eof.SpatioTemporalPCA(ds[var], n_components=n_components),
        )
    pca_coll = eof.PCACollection(pca_lst)
    n_components = pca_coll.n_components
    pcs = pca_coll.get_principal_components()

    # Split in training and val data
    idx_train = int(0.75*len(pcs['time']))
    idx_val = int(0.9*len(pcs['time'])) 
    data = dict(
        train=pcs.isel(time=slice(None,idx_train)),
        val=pcs.isel(time=slice(idx_train, idx_val)),
        test=pcs.isel(time=slice(idx_val, None))
    )

    # In case LIM is only trained on a subset of the PCs
    if n_eof_lim is not None:
        assert n_eof_lim < n_eof
        data4lim = {}
        for key in data.keys():
            pcs_truncate = []
            n_start = 0
            for i, pca in enumerate(pca_coll.pca_lst):
                n_end = n_start + (n_eof_lim // (i+1))
                pcs_truncate.append(data[key].isel(eof=slice(n_start, n_end)))
                n_start = pca.n_components
            data4lim[key] = xr.concat(pcs_truncate, dim='eof')
    else:
        data4lim = data.copy()

    # Fit LIM
    # ======================================================================================
    frcst_lim = {}
    if lim_type == 'stlim':
        # Stationary LIM
        print("Create ST-LIM forecast!", flush=True)
        tau = 2
        model = lim.LIM(tau)
        model.fit(data4lim['train'].data)

        # Create deterministic forecasts 
        lag_arr = np.arange(1, horiz+1, 1)
        for key, z in data4lim.items():
            frcst_lim[key] = xr.DataArray(
                data=model.rollout_mean(z.data, lag_arr), 
                coords={'lag': lag_arr, 'eof': z['eof'],
                        'time': z['time']}
        ).transpose('eof', 'time', 'lag')

    elif lim_type == 'cslim':
        # Cyclostationary LIM
        print("Create CS-LIM forecast!", flush=True)
        tau = 1
        model = lim.CSLIM(tau=tau)
        start_month = data4lim['train'].time.dt.month[0].data
        model.fit(data4lim['train'].data, start_month, average_window=3)

        # Create deterministic forecasts 
        lag_arr = np.arange(1, horiz+1, 1)
        for key, z in data4lim.items():
            rollout_arr = []
            for i in range(len(z['time'])):
                z_init = z.isel(time=i)
                month = z_init.time.dt.month.data
                z_rollout = model.rollout_mean(z_init.data, month, lag_arr)
                rollout_arr.append(z_rollout)
                
            frcst_lim[key] =  xr.DataArray(
                data=np.array(rollout_arr),
                coords={'time': z['time'], 'lag': lag_arr, 'eof': z['eof']}
                ).transpose('eof', 'time', 'lag')
    else:
        raise ValueError(f"Specified lim_type={lim_type} does not exist!")

    # In case LIM forecast has only been applied on subset of PCs, add zeros
    # to all remaining PCs 
    if n_eof_lim is not None:
        for key, z_lim in frcst_lim.items():
            buff = []
            n_start = 0
            idx_start_eof = 0
            pad_axis = np.where(np.array(z_lim.dims)== 'eof')
            for i, pca in enumerate(pca_coll.pca_lst):
                n_end = n_start + (n_eof_lim // (i+1))
                n_pad = pca.n_components - (n_eof_lim // (i+1))
                pad_width = np.zeros((len(z_lim.dims), 2), dtype=int)
                pad_width[pad_axis,1] = n_pad
                buff.append(
                    xr.DataArray(
                        data=np.pad(z_lim.isel(eof=slice(n_start, n_end)).data,
                                    pad_width=tuple(map(tuple,pad_width)),
                                    mode='constant'),
                        coords=dict(eof=np.arange(idx_start_eof, idx_start_eof+pca.n_components+1),
                                    time=z_lim['time'], lag=z_lim['lag'])
                     )
                )
                n_start = n_end
                idx_start_eof += pca.n_components

            frcst_lim[key] = xr.concat(buff, dim='eof')
    
    # Normalize PCs 
    # ======================================================================================
    print("Normalize PC for LSTM training.", flush=True)
    normalizer_pca = preproc.Normalizer(method='zscore')
    normalizer_pca.fit(pcs, dim='time')

    # Normalize target data 
    for key, z in data.items():
        data[key] = normalizer_pca.transform(z)

    # Normalize LIM forecast
    for key, z in frcst_lim.items():
        frcst_lim[key] = normalizer_pca.transform(z)
    
    
    # Torch dataset class
    # ======================================================================================
    transform = transforms.Compose([ToTensor()])

    datasets = {}
    dataloaders = {}
    for key in data.keys():
        datasets[key] = TimeSeriesResidual(data[key], frcst_lim[key], None, 
                                           transform=transform)
        shuffle = True if key == 'train' else False
        dataloaders[key] = DataLoader(datasets[key],
                                      drop_last=True,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=shuffle)
                                

    return ds, datasets, dataloaders, pca_coll, normalizer_pca 


def load_pcdata_lim_ensemble(
    lim_hindcast: xr.Dataset, datapaths: dict = None, 
    n_eof: int=20, num_traindata: int=None,
    batch_size: int=32,
    **kwargs
    ) -> list:
    """Load data for LIM+LSTM model with LIM ensemble prediction as input.

    Args:
        datapaths (dict): _description_
        lim_hindcast (xr.Dataset): LIM hindcast with members.
        hist (int, optional): _description_. Defaults to 0.
        horiz (int, optional): _description_. Defaults to 24.
        n_eof (int, optional): Number of eofs. Defaults to 20.
        num_traindata (int, optional): _description_. Defaults to None.
        batch_size (int, optional): _description_. Defaults to 32.

    Returns:
        ds (xr.Dataset): 
        datasets (hyblim.ResidualTimeSeries):
        dataloaders (torch.Dataloader):
        pca_coll (hyblim.MultiSpatioTemporalPCA):
        normalizer_pca (hyblim.Normalizer):
    """
    # Load preprocessed data
    print("Load data!", flush=True)
    da_arr = []
    for var, path in datapaths.items():
        da = xr.open_dataset(path)[var]
        # Create normalizer from stored attributes
        if 'method' in da.attrs:
            normalizer = preproc.normalizer_from_dict(da.attrs)
            da.attrs['normalizer'] = normalizer
        da_arr.append(da)

    ds = xr.merge(da_arr) 

    # Create PCA
    # ======================================================================================
    pca_lst = []
    for i, var in enumerate(ds.data_vars):
        print(f"Create EOF of {var}!")
        n_components = n_eof // (i+1)
        pca_lst.append(
            eof.SpatioTemporalPCA(ds[var], n_components=n_components),
        )
    pca_coll = eof.PCACollection(pca_lst)
    n_components = pca_coll.n_components
    pcs = pca_coll.get_principal_components()

    # Normalize PCs 
    # ======================================================================================
    print("Normalize PC for LSTM training.", flush=True)
    normalizer_pca = preproc.Normalizer(method='zscore')
    normalizer_pca.fit(pcs, dim='time')

    # Normalize target
    target = normalizer_pca.transform(pcs)

    # Normalize LIM hindcast
    lim_hindcast = normalizer_pca.transform(lim_hindcast)

    # Split in training and val data
    # ======================================================================================
    # Split in training and val data
    idx_val = int(0.75*len(pcs['time']))
    idx_test = int(0.9*len(pcs['time'])) 
    if num_traindata is None:
        idx_train = int(0.75*len(pcs['time']))
        idx_val_lim = idx_val 
        idx_test_lim = idx_test 
    else:
        idx_train = num_traindata
        idx_val_lim = idx_train 
        idx_test_lim = idx_train + (idx_test-idx_val)

    target = dict(
        train=target.isel(time=slice(None,idx_train)),
        val=target.isel(time=slice(idx_val, idx_test)),
        test=target.isel(time=slice(idx_test, None))
    )
    lim_hindcast = dict(
        train=lim_hindcast.isel(time=slice(None,idx_train)),
        val=lim_hindcast.isel(time=slice(idx_val_lim, idx_test_lim)),
        test=lim_hindcast.isel(time=slice(idx_test_lim, None))
    )

    # Check if hindcast has same dimensions than target
    # ======================================================================================
    print("Check if hindcast and target have same coordinates!", flush=True)
    assert np.array_equal(target['train']['time'].data, lim_hindcast['train']['time'].data)
    assert np.array_equal(target['train']['eof'].data, lim_hindcast['train']['eof'].data)

    # Torch dataset class
    # ======================================================================================
    transform = transforms.Compose([ToTensor()])

    datasets = {}
    dataloaders = {}
    for key in target.keys():
        datasets[key] = TimeSeriesResidual(target[key], lim_hindcast[key],
                                           transform=transform)
        shuffle = True if key == 'train' else False
        dataloaders[key] = DataLoader(datasets[key],
                                      drop_last=True,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=shuffle)

    return ds, datasets, dataloaders, pca_coll, normalizer_pca 


def load_stdata_lim(
    path_target: str, paths_lim_frcst: list,
    hist: int=0, batch_size: int=32, testing=False,
    **kwargs
    ) -> list:
    """Load gridded data and the LIM-hindcast to create SpatioTemporalResidual dataset
    and dataloader. 

    Args:
        path_target (str): _description_
        paths_lim_frcst (list): _description_
        hist (int, optional): _description_. Defaults to 0.
        batch_size (int, optional): _description_. Defaults to 32.
        testing (bool, optional): _description_. Defaults to False.

    Returns:
        ds_target (xr.Dataset)
        ds_frcst_lim (xr.Dataset)
        datasets (dict): Dict of SpatioTemporalResidual
        dataloaders (dict): Dataloaders of ['train', 'val', 'test']
    """

    # Load data
    if testing: # True only for testing
        print("Load only a subset of the data for testing.", flush=True)
        ds_target = xr.open_dataset(path_target).isel(time=slice(None,500)) 
        ds_frcst_lim = xr.open_mfdataset(
            paths_lim_frcst, combine='nested', concat_dim='lag'
        ).isel(time=slice(None,500))
    else:
        print("Load data!", flush=True)
        ds_target = xr.open_dataset(path_target)  
        ds_frcst_lim = xr.open_mfdataset(
            paths_lim_frcst, combine='nested', concat_dim='lag'
        )

    # Create normalizer from stored parameters
    for var in ds_target.data_vars:
        normalizer = preproc.normalizer_from_dict(ds_target[var].attrs)
        ds_target[var].attrs['normalizer'] = normalizer
        ds_frcst_lim[var].attrs['normalizer'] = normalizer

    # Split in training and val data
    idx_train = int(0.75*len(ds_target['time']))
    idx_val = int(0.9*len(ds_target['time'])) 

    data = dict(
        train=ds_target.isel(time=slice(None,idx_train)),
        val=ds_target.isel(time=slice(idx_train, idx_val)),
        test=ds_target.isel(time=slice(idx_val, None))
    )
    frcst_lim = dict(
        train=ds_frcst_lim.isel(time=slice(None,idx_train)),
        val=ds_frcst_lim.isel(time=slice(idx_train, idx_val)),
        test=ds_frcst_lim.isel(time=slice(idx_val, None))
    )
    # Create torch dataset
    # ======================================================================================
    print("Create datasets!")
    transform = transforms.Compose([ForConvolution(), ToTensor()])

    datasets = {}
    dataloaders = {}
    for key in data.keys():
        datasets[key] = SpatioTemporalResidual(data[key], frcst_lim[key], 
                                           hist=hist, transform=transform)
        shuffle = True if key == 'train' else False
        dataloaders[key] = DataLoader(datasets[key],
                                      drop_last=True,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=shuffle)


    return ds_target, ds_frcst_lim, datasets, dataloaders