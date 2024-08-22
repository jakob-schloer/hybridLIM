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


def _to_tensor(sample: dict):
    """ Impute NaNs and convert numpy arrays to torch tensors."""
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            value = np.nan_to_num(value, copy=False, nan=0.0)
            sample[key] = torch.from_numpy(value).float()
        elif isinstance(value, int):
            sample[key] = torch.tensor(value).float()
        elif isinstance(value, list):
            sample[key] = torch.tensor(value).float()
        else:
            continue
    return sample


class TimeSeries(Dataset):
    def __init__(self, data, n_timesteps, transform=None):
        self.data = data.compute()
        self.n_timesteps = n_timesteps
        self.transform = transform
    
    def __len__(self):
        return len(self.data['time']) - self.n_timesteps - 1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_interval = self.data.isel(time=slice(idx, idx+self.n_timesteps))
        month = x_interval['time'].dt.month.data.astype(int) - 1
        
        sample = _to_tensor({
            'x':   x_interval.data,
            'time': preproc.time2timestamp(x_interval['time'].data),
            'month': month,
            'idx': np.arange(idx, idx+self.n_timesteps)
        })

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
        self.data = target.compute()
        self.lim_ensemble = lim_ensemble.compute() 
        self.lag_arr = lim_ensemble['lag'].values
        self.transform = transform

        assert len(self.data['time']) == len(self.lim_ensemble['time'])


    def __len__(self):
        return len(self.lim_ensemble['time']) - np.max(self.lag_arr)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {}
        sample['lim'] = self.lim_ensemble.isel(time=idx).data
        ids_target = idx + self.lag_arr
        sample['idx'] = ids_target
        sample['target'] = self.data.isel(time=ids_target).data
        sample['month'] = self.data['time'][ids_target].dt.month.data.astype(int) - 1
        sample['time'] = preproc.time2timestamp(
                self.data.isel(time=ids_target)['time'].data
        )
        sample['lag'] = self.lag_arr

        sample = _to_tensor(sample)
        if self.transform:
            sample = self.transform(sample)

        return sample['lim'], sample['target'], {'time': sample['time'], 'idx': sample['idx'],
                                                 'month': sample['month'], 'lag': sample['lag']}


class SpatialTemporalData(Dataset):
    """Spatial temporal data of multible variables.. 

    Args:
        data (xr.Dataset): Dataset with dimensions ['time', 'lat', 'lon'].
        n_timesteps (int): Length of time-series. 
        transform ([type], optional): [description]. Defaults to None.
    """

    def __init__(self, data: xr.Dataset, n_timesteps: int,
                 transform=None):
        super().__init__()

        self.data = data.transpose('time', 'lat', 'lon').compute()
        self.vars = list(self.data.data_vars)
        self.coords = list(self.data.coords)
        self.dims = self.data[self.vars[0]].shape
        self.n_timesteps = n_timesteps
        
        self.transform = transform
    

    def __len__(self)-> int: 
        """Number of samples in dataset."""
        return len(self.data['time']) - self.n_timesteps - 1
    

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        x_interval = self.data.isel(time=slice(idx, idx+self.n_timesteps))
        
        sample = _to_tensor({
            'data':  np.array([x_interval[var].data for var in self.vars]),
            'time':  preproc.time2timestamp(self.data['time'].data[idx]),
            'month': x_interval['time'].dt.month.data.astype(int) - 1,
            'idx': np.arange(idx, idx+self.n_timesteps)
        })
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
                          'lat': self.data.coords['lat'],
                          'lon': self.data.coords['lon']}

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
                              'lat': self.data.coords['lat'],
                              'lon': self.data.coords['lon']}

                    da_arr.append(xr.DataArray(
                        data=data_map,
                        coords=coords,
                        name=var
                    ))

                ds.append(xr.merge(da_arr))

        return ds


# ======================================================================================
# Functions to load data for each class  
# ======================================================================================
def load_stdata(datapaths: dict, lsm_path: str,
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
    # Load data
    # ======================================================================================
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
    lsm = xr.open_dataset(lsm_path)['lsm']
    ds = ds.where(lsm!=1, other=np.nan)

    # Split in training and test data
    # ======================================================================================
    if num_traindata is None:
        train_period = (0, int(0.8*len(ds['time'])))
    else:
        idx_start = np.random.randint(0, int(0.8*len(ds['time'])) - num_traindata)
        train_period = (idx_start, idx_start + num_traindata)
    val_period = (int(0.8*len(ds['time'])), int(0.9*len(ds['time'])))
    test_period = (int(0.9*len(ds['time'])), len(ds['time']))

    data = dict(
        train = ds.isel(time=slice(*train_period)),
        val = ds.isel(time=slice(*val_period)),
        test = ds.isel(time=slice(*test_period))
    )

    # Torch dataset class
    # ======================================================================================
    n_timesteps = hist + horiz

    datasets, dataloaders = {}, {}
    for key, values in data.items():
        datasets[key] = SpatialTemporalData(values, n_timesteps)
        shuffle = True if key == 'train' else False
        dataloaders[key] = DataLoader(datasets[key],
                                      drop_last = True,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=shuffle)

    return ds, datasets, dataloaders 


def load_pcdata(
    datapaths: dict, lsm_path: str, 
    n_eof: list=[20, 10], hist: int=4, horiz: int=24, 
    num_traindata: int=None, batch_size: int=32,
    **kwargs) -> list:
    """Load dataset for training LSTM on PCs.

    Args:
        datapaths (dict): Dictionaries with datasets,
        lsm_path (str): Path to land sea mask.
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
    # Load data
    # ======================================================================================
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
    lsm = xr.open_dataset(lsm_path)['lsm']
    ds = ds.where(lsm!=1, other=np.nan)

    # Create PCA
    # ======================================================================================
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

    # Normalize PCs
    z_eof = combined_eof.transform(ds)
    print("Normalize PCs.", flush=True)
    normalizer_pca = preproc.Normalizer(method='zscore')
    z_eof_norm = normalizer_pca.fit_transform(z_eof, dim='time')

    # Split in training and test data
    # ======================================================================================
    if num_traindata is None:
        train_period = (0, int(0.8*len(ds['time'])))
    else:
        idx_start = np.random.randint(0, int(0.8*len(ds['time'])) - num_traindata)
        train_period = (idx_start, idx_start + num_traindata)
    val_period = (int(0.8*len(ds['time'])), int(0.9*len(ds['time'])))
    test_period = (int(0.9*len(ds['time'])), len(ds['time'])) 

    data = dict(
        train = z_eof_norm.isel(time=slice(*train_period)),
        val = z_eof_norm.isel(time=slice(*val_period)),
        test = z_eof_norm.isel(time=slice(*test_period)),
    )

    # Torch dataset class
    # ======================================================================================
    n_timesteps = hist + horiz

    datasets = {}
    dataloaders = {}
    for key, values in data.items():
        datasets[key] = TimeSeries(values,n_timesteps)
        shuffle = True if key == 'train' else False
        dataloaders[key] = DataLoader(datasets[key],
                                      drop_last = True,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=shuffle)

    return ds, datasets, dataloaders, combined_eof, normalizer_pca  



def load_pcdata_lim_ensemble(
    lim_hindcast: xr.Dataset, 
    datapaths: dict = None, 
    lsm_path: str = None,
    n_eof: list=[20, 10],
    num_traindata: int=None,
    batch_size: int=32,
    **kwargs
    ) -> list:
    """Load data for LIM+LSTM model with LIM ensemble prediction as input.

    Args:
        datapaths (dict): _description_
        lim_hindcast (dict): Dictionary of hindcasts with "train", "val", "test" data.
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
    # Load data
    # ======================================================================================
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
    lsm = xr.open_dataset(lsm_path)['lsm']
    ds = ds.where(lsm!=1, other=np.nan)

    # Create PCA
    # ======================================================================================
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

    # Normalize PCs
    z_eof = combined_eof.transform(ds)
    print("Normalize PCs.", flush=True)
    normalizer_pca = preproc.Normalizer(method='zscore')
    z_eof_norm = normalizer_pca.fit_transform(z_eof, dim='time')

    # Normalize LIM hindcast
    for key, z_lim in lim_hindcast.items():
        lim_hindcast[key] = normalizer_pca.transform(z_lim)

    # Split in training and test data
    # ======================================================================================
    target = dict()
    for key in ['train', 'val', 'test']:
        times = lim_hindcast[key]['time'].data
        target[key] = z_eof_norm.sel(time=times)

    assert np.array_equal(target['train']['eof'].data, lim_hindcast['train']['eof'].data)

    # Torch dataset class
    # ======================================================================================
    datasets = {}
    dataloaders = {}
    for key in target.keys():
        datasets[key] = TimeSeriesResidual(target[key], lim_hindcast[key])
        shuffle = True if key == 'train' else False
        dataloaders[key] = DataLoader(datasets[key],
                                      drop_last=True,
                                      batch_size=batch_size,
                                      num_workers=4,
                                      pin_memory=True,
                                      shuffle=shuffle)

    return ds, datasets, dataloaders, combined_eof, normalizer_pca 

