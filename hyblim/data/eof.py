"""Empirical Orthogonal Function (EOF) analysis."""
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from hyblim.data import preproc
 

class EmpiricalOrthogonalFunctionAnalysis:
    """Empirical Orthogonal Function Analysis (EOFA) is pca of spatio-temporal data.

    Wrapper for sklearn.decomposition.PCA that works with xr.Datasets of dimension (time, lat, lon).

    Args:
        n_components (int): Number of components for PCA
    """
    def __init__(self, n_components: int, **kwargs):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components, **kwargs)
        self.ids_notNaN = None


    def fit(self, ds: xr.Dataset) -> None:
        """Fit PCA to data.
        
        Args:
            ds (xr.Dataset): Input data to perform PCA on. If dataset contains multiple variables,
                the PCA is performed on the concatenated data (multivariate PCA). The dataset dimensions
                have to be (time, 'lat', 'lon').
        """
        X, self.ids_notNaN = preproc.map2flatten(ds)
        self.pca.fit(X.data)
        return None


    def transform(self, ds: xr.Dataset) -> xr.DataArray:
        """Returns time evolution of components.

        Args:
            ds (xr.Dataset): Input dataarray to perform PCA on.
                Array dimensions (time, 'lat', 'lon')

        Return:
            z (xr.Dataarray): Transformed data of shape (n_components, time)
        """
        X_transform, ids_notNaN = preproc.map2flatten(ds)
        assert np.count_nonzero(ids_notNaN) == np.count_nonzero(self.ids_notNaN)
        
        z = []
        for i, comp in enumerate(self.pca.components_):
            z_comp = X_transform.data @ comp

            da_z_comp = xr.DataArray(
                data=z_comp,
                dims=['time'],
                coords=dict(time=X_transform['time']),
            )
            z.append(da_z_comp)
        return xr.concat(z, dim='eof')
    

    def fit_transform(self, ds: xr.Dataset) -> xr.DataArray:
        """Fit and transform data.

        Args:
            ds (xr.Dataset): Input dataarray to perform PCA on.
                Array dimensions (time, 'lat', 'lon')

        Return:
            z (xr.Dataarray): Transformed data of shape (n_components, time)
        """
        assert self.pca.components_ is not None
        self.fit(ds)
        return self.transform(ds)


    def components(self):
        """Return components of EOFA.

        Return:
            components (xr.dataarray): Size (n_components, N_x, N_y)
        """
        # EOF maps
        eof_map = []
        for i, comp in enumerate(self.pca.components_):
            eof = preproc.flattened2map(comp, self.ids_notNaN)
            eof_map.append(eof)

        return xr.concat(eof_map, dim='eof')
    

    def explained_variance(self):
        return self.pca.explained_variance_ratio_


    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct pc-space to data space

        Args:
            z (np.ndarray): Time series in PC space of shape (n_times, n_eof) 

        Returns:
            x (np.ndarray): Time series in data space of shape (n_times, n_data) 
        """
        x = z @ self.pca.components_
        return x
    

    def reconstruction(self, z: np.ndarray, newdim: str='x') -> xr.Dataset:
        """Reconstruct the dataset from components and time-evolution.

        Args:
            z (np.ndarray): Low dimensional vector of size (n_time, n_components)
            newdim (str, optional): Name of dimension. Defaults to 'x'.

        Returns:
            xr.Dataset: Reconstructed data of shape (lat, lon, n_time)
        """
        x_flatten = self.inverse_transform(z)

        x_map_list = []
        for x in x_flatten:
            x_map = preproc.flattened2map(x, self.ids_notNaN)
            x_map_list.append(x_map)

        return xr.concat(x_map_list, dim=newdim)
    
    
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
    

class CombinedEOF:
    """ Combine EOFs of different variables into one object, i.e. [EOF1, EOF2, ...] -> CombinedEOF.

    Args:
        eof_lst (list): List of EOFs to be combined.
        vars (list): Labels of the EOFs, e.g. ['ssta', 'ssha'].
    """
    def __init__(self, eofa_lst: list, vars: list) -> None:
        self.eofa_lst = eofa_lst
        self.vars = vars

        self.n_components = 0
        for eofa in self.eofa_lst:
            self.n_components += eofa.n_components

    def components(self) -> xr.DataArray:
        """Return components of the combined EOFs.

        Returns:
            components (xr.DataArray): Components of the combined EOFs.
        """
        components = []
        n_start = 1
        for i, eofa in enumerate(self.eofa_lst):
            comp = eofa.components()
            n_end = n_start + len(comp['eof'])
            comp = comp.assign_coords(dict(
                eof=np.arange(n_start, n_end), 
                var=('eof', [self.vars[i]]*len(comp['eof'])), 
            ))
            components.append(comp)
            n_start = n_end 
        return xr.concat(components, dim='eof')
    
    def explained_variance(self):
        explained_variance = []
        for eofa in self.eofa_lst:
            explained_variance.append(eofa.explained_variance())
        return np.concatenate(explained_variance)

    def transform(self, ds: xr.Dataset) -> xr.DataArray:
        """Transform to PCs.

        Args:
            ds (xr.Dataset): Input to be projected on EOFs with dimension ('time', ...) 

        Returns:
            z (xr.Dataarray): Transformed data of shape (n_components, time).
        """
        n_times = ds.dims['time']
        z = np.zeros((n_times, self.n_components))
        n_start = 0
        for i, var in enumerate(ds.data_vars):
            eofa = self.eofa_lst[i]
            n_end = n_start + eofa.n_components
            z[:, n_start:n_end] = eofa.transform(ds[var]).data.T
            n_start = n_end

        return xr.DataArray(z, coords=dict(time=ds['time'], eof=np.arange(1, self.n_components+1)))
    

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct pc-space to data space

        Args:
            z (np.ndarray): Time series in PC space of shape (n_times, n_eof) 

        Returns:
            x (np.ndarray): Time series in data space of shapte (n_times, n_data) 
        """
        x = []
        n_start = 0
        for eofa in self.eofa_lst:
            n_end = n_start + eofa.n_components
            x.append(eofa.inverse_transform(z[:, n_start:n_end]))
            n_start = n_end

        return np.concatenate(x, axis=1)
    
    def reconstruction(self, z: np.ndarray, times: np.ndarray=None) -> xr.Dataset:
        """Transform PC to data space.

        Args:
            z (np.ndarray): PC time series of shape(n_times, n_eof)
            times (np.ndarray): Time coordinate of shape (n_times)

        Returns:
            xr.Dataset: Transformed back to dataspace. Shape (lat, lon, n_times) 
        """
        assert z.shape[0] == len(times)
        x_reconstructed = []
        n_start = 0
        for i, eofa in enumerate(self.eofa_lst):
            n_end = n_start + eofa.n_components
            x_map = eofa.reconstruction(z[:, n_start:n_end], newdim=pd.Index(times, name='time'))
            x_map.name = self.vars[i]
            x_reconstructed.append(x_map)
            n_start = n_end

        return xr.merge(x_reconstructed) 
    

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
        for eofa in self.eofa_lst:
            n_end = n_start + eofa.n_components
            C_eof_var = C_eof[n_start: n_end, n_start: n_end]
            A = eofa.pca.compontents
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
        for eofa in self.eofa_lst:
            n_end = n_start + eofa.n_components
            # Std in eof space
            var_z = std_eof[:, n_start: n_end]**2
            # Transfomation matrix: A
            A = eofa.pca.components_
            # A*std for each time
            scaled_A = A*var_z[:,:,np.newaxis]
            # A*std @ A.T for each time
            var_x = np.einsum('ijk,jk->ik', scaled_A, A)

            var_data.append(var_x)
            n_start = n_end

        var_data = np.concatenate(var_data, axis=1)

        return np.sqrt(var_data)
     

# ======================================================================================
# Helper functions
# ======================================================================================
def inverse_transform_of_random_latent(eofa: EmpiricalOrthogonalFunctionAnalysis, 
                                       ignore_n_components: int,
                                       n_time: int) -> np.ndarray:
    """Add variance from higher EOFs.
    
    Randomly create PCs > n_predicted_components and transfrom to grid space.

    Args:
        eofa (EmpiricalOrhtogonalFunctionAnalysis): PCA with for instance 300 components.
        ignore_n_components (int): Ignore the first n components.
        n_time (int): Number of samples.
    Returns:
        x_rand (np.ndarray): Randomly sampled pcs projected to grid space
    """
    # Sample random white noise with right scale
    explained_var = eofa.explained_variance()[ignore_n_components:]
    z_rand = np.random.normal(size=(n_time, len(explained_var))) * np.sqrt(explained_var) 

    # Transform to flattened grid space 
    x_rand = z_rand @ eofa.pca.components_[ignore_n_components:, :]

    return x_rand