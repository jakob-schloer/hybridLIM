"""PCA for spatio-temporal data."""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from hyblim.data import preproc
 

class SpatioTemporalPCA:
    """PCA of spatio-temporal data.

    Wrapper for sklearn.decomposition.PCA with xarray.DataArray input.

    Args:
        ds (xr.DataArray or xr.Dataset): Input dataarray to perform PCA on.
            Array dimensions (time, 'lat', 'lon')
        n_components (int): Number of components for PCA
    """
    def __init__(self, ds, n_components, **kwargs):
        self.ds = ds

        self.X, self.ids_notNaN = preproc.map2flatten(self.ds)

        # PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(self.X.data)

        self.n_components = self.pca.n_components


    def get_eofs(self):
        """Return components of PCA.

        Return:
            components (xr.dataarray): Size (n_components, N_x, N_y)
        """
        # EOF maps
        eof_map = []
        for i, comp in enumerate(self.pca.components_):
            eof = preproc.flattened2map(comp, self.ids_notNaN)
            eof_map.append(eof)

        return xr.concat(eof_map, dim='eof')
    

    def get_principal_components(self):
        """Returns time evolution of components.

        Return:
            time_evolution (xr.Dataarray): Principal components of shape (n_components, time)
        """
        time_evolution = []
        for i, comp in enumerate(self.pca.components_):
            ts = self.X.data @ comp

            da_ts = xr.DataArray(
                data=ts,
                dims=['time'],
                coords=dict(time=self.X['time']),
            )
            time_evolution.append(da_ts)
        return xr.concat(time_evolution, dim='eof')
    

    def get_explainedVariance(self):
        return self.pca.explained_variance_ratio_


    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct pc-space to data space

        Args:
            z (np.ndarray): Time series in PC space of shape (n_times, n_eof) 

        Returns:
            x (np.ndarray): Time series in data space of shapte (n_times, n_data) 
        """
        x = z @ self.pca.components_
        return x
    
    
    def inverse_transform_std(self, std_eof: np.ndarray) -> list:
        """Transform standard deviation in eof space to std in data space.

        Args:
            std_eof (np.ndarray): Std in eof space of shape (n_times, n_components) 

        Returns:
            std_data (list): Std ind data space of shape (n_times, n_data)
        """
        # Transfomation matrix: A
        A = self.pca.components_
        # A*std for each time
        scaled_A = A*std_eof[:,:,np.newaxis]
        # A*std @ A.T for each time
        std_data = np.einsum('ijk,jk->ik', scaled_A, A)

        return std_data
    
    
    def inverse_transform_cov(self, Cz: np.ndarray) -> list:
        """Transform the covariance in eof space to data space.

        Cx = A @ Cz @ A.T

        Args:
            Cz (np.ndarray): Covariance of shape (n_components, n_components)

        Returns:
            Cx (np.ndarray): Covariances in data space of shape (n_data, n_data). 
        """
        A = self.pca.compontents
        Cx = A.T @ Cz @ A 

        return Cx
    

    def reconstruction(self, z, newdim='x'):
        """Reconstruct the dataset from components and time-evolution.

        Args:
            z (np.ndarray): Low dimensional vector of size (time, n_components)
            newdim (str, optional): Name of dimension. Defaults to 'x'.

        Returns:
            _type_: _description_
        """
        reconstruction = z.T @ self.pca.components_

        rec_map = []
        for rec in reconstruction:
            x = preproc.flattened2map(rec, self.ids_notNaN)
            rec_map.append(x)

        rec_map = xr.concat(rec_map, dim=newdim)

        return rec_map


class PCACollection:

    def __init__(self, pca_lst: list) -> None:
        self.pca_lst = pca_lst

        # Data
        ds_lst = []
        for pca in self.pca_lst:
            ds_lst.append(pca.ds)
        ds_merge = xr.merge(ds_lst)

        self.X, self.ids_notNaN = preproc.map2flatten(ds_merge)

        self.n_components = 0
        for pca in self.pca_lst:
            self.n_components += pca.pca.n_components

    def get_eofs(self):
        eofs = []
        n_start = 1
        for pca in self.pca_lst:
            eof_var = pca.get_eofs()
            n_end = n_start + len(eof_var['eof'])
            eof_var = eof_var.assign_coords(dict(
                eof=np.arange(n_start, n_end ), 
                var=('eof', [pca.X.name]*len(eof_var['eof'])), 
            ))
            eofs.append(eof_var)
            n_start = n_end 
        return xr.concat(eofs, dim='eof')


    def get_principal_components(self):
        pcs = []
        n_start = 1
        for pca in self.pca_lst:
            pc_var = pca.get_principal_components()
            n_end = n_start + len(pc_var['eof'])
            pc_var = pc_var.assign_coords(dict(
                eof=np.arange(n_start, n_end ), 
                var=('eof', [pca.X.name]*len(pc_var['eof'])), 
            ))
            pcs.append(pc_var)
            n_start = n_end 
        return xr.concat(pcs, dim='eof')
    

    def explained_variance(self):
        explained_variance = []
        for pca in self.pca_lst:
            explained_variance.append(pca.get_explainedVariance())
        explained_variance = np.concatenate(explained_variance)

        return explained_variance


    def tranform(self, ds: xr.Dataset):
        """Transform to PCs.

        Args:
            ds (xr.Dataset): Input to be projected on EOFs with dimension ('time', ...) 

        Returns:
            z (xr.Dataarray): PCs.
        """
        n_times = ds.dims['time']
        z = np.zeros((n_times, self.n_components))
        n_start = 0
        for i, var in enumerate(ds.data_vars):
            x_flat, ids_nan = preproc.map2flatten(ds[var])
            pca = self.pca_lst[i]
            assert np.count_nonzero(ids_nan) == np.count_nonzero(pca.ids_notNaN)
            n_end = n_start + pca.pca.n_components
            z[:,n_start:n_end] = pca.pca.transform(x_flat)
            n_start = n_end

        z = xr.DataArray(z, coords=dict(time=ds['time'], eof=np.arange(self.n_components)))
        return z
    

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct pc-space to data space

        Args:
            z (np.ndarray): Time series in PC space of shape (n_times, n_eof) 

        Returns:
            x (np.ndarray): Time series in data space of shapte (n_times, n_data) 
        """
        x = []
        n_start = 0
        for pca in self.pca_lst:
            n_end = n_start + pca.pca.n_components
            x.append(z[:, n_start:n_end] @ pca.pca.components_)
            n_start = n_end

        x = np.concatenate(x, axis=1)
        assert x.shape[0] == z.shape[0]
        return x

    
    def inverse_transform_covariance(self, C_eof: np.ndarray) -> list:
        """Transform the covariance/ covariances in stacked eof space
            into data space.

        Args:
            C_eof (np.ndarray): Covariance of shape 
                (n_components, n_components)

        Returns:
            C_data (list): List of covariances in data space. 
        """
        C_data = [] 
        n_start = 0
        for pca in self.pca_lst:
            n_end = n_start + pca.n_components
            C_eof_var = C_eof[n_start: n_end, n_start: n_end]
            A = pca.pca.compontents
            C_data.append(
                A.T @ C_eof_var @ A 
            )
            n_start = n_end
        return C_data
    
    
    def inverse_transform_std(self, std_eof: np.ndarray) -> list:
        """Transform standard deviation in eof space to std in data space.

        Args:
            std_eof (np.ndarray): Std in eof space of shape (n_times, n_components) 

        Returns:
            std_data (list): Std ind data space of shape (n_times, n_data)
        """
        var_data = [] 
        n_start = 0
        for stpca in self.pca_lst:
            n_end = n_start + stpca.n_components
            # Std in eof space
            var_z = std_eof[:, n_start: n_end]**2
            # Transfomation matrix: A
            A = stpca.pca.components_
            # A*std for each time
            scaled_A = A*var_z[:,:,np.newaxis]
            # A*std @ A.T for each time
            var_x = np.einsum('ijk,jk->ik', scaled_A, A)

            var_data.append(var_x)
            n_start = n_end

        var_data = np.concatenate(var_data, axis=1)

        return np.sqrt(var_data)

    

    def reconstruction(self, z, times=None):
        """Transform PC to data space.

        Args:
            z (np.ndarray): PC time series of shape(n_times, n_eof)
            times (np.ndarray): Time coordinate of shape (n_times)

        Returns:
            xr.Dataset: Transformed back to dataspace. Shape (lat, lon, n_times) 
        """
        x_flat = self.inverse_transform(z)
        # Transform to map
        x_map = preproc.flattened2map(x_flat, self.ids_notNaN, times)

        return x_map
    

# ======================================================================================
# Helper functions
# ======================================================================================
def inverse_transform_white_noise_pcs(extended_pca: PCACollection, 
                                      n_predicted_components: list,
                                      n_time: int) -> np.ndarray:
    """Add variance from higher EOFs.
    
    Randomly create PCs > n_predicted_components and transfrom to grid space.

    Args:
        extended_pca (eof.PCACollection): PCA with for instance 300 components.
        n_predicted_components (list): Predicted components which are not random.
        n_time (int): Number of samples.

    Returns:
        np.ndarray: _description_
    """
    n_components_full = extended_pca.pca_lst[0].pca.n_components

    x_rand = []
    for i, pca in enumerate(extended_pca.pca_lst):
        n_start = n_predicted_components[i] + i * n_components_full 
        n_end = (i+1) * n_components_full 
        # Sample random white noise with right scale
        explained_var = pca.pca.explained_variance_[n_start:n_end]
        z_rand = np.random.normal(
            size=(n_time, len(explained_var))) * np.sqrt(explained_var) 

        # Transform to flattened grid space 
        U = pca.pca.components_[n_start:n_end, :]
        x_rand.append(z_rand @ U)

    return np.concatenate(x_rand, axis=1)