''' File description

@Author  :   Jakob Schlör 
@Time    :   2022/10/05 08:49:58
@Contact :   jakob.schloer@uni-tuebingen.de
'''
import os, pickle, functools, cftime
from re import I
import numpy as np
import scipy.linalg as linalg
import pandas as pd
import xarray as xr

from hyblim.data import preproc


def matrix_decomposition(A):
    """Eigendecomposition of square matrix:
    
        A = U D V.T
    Args:
        A (np.ndarray): Square matrix 

    Returns:
        w (np.ndarray): Sorted eigenvalues
        U (np.ndarray): Matrix of sorted eigenvectors of A
        V (np.ndarray): Matrix of sorted eigenvectors of A.T 
    """
    w, U = np.linalg.eig(A)
    idx_sort = np.argsort(w)[::-1]
    w = w[idx_sort]
    U = U[:, idx_sort]

    w_transpose, V = np.linalg.eig(A.T)
    idx_sort = np.argsort(w_transpose)[::-1]
    V = V[:, idx_sort]

    return w, U, V


def integrate_lim(L, noise_cov, x_0=None, n_times=100, dt=1):
    """Integration of LIM by Penland 1994.

    Y(t+dt) = Y(t) + L @ Y(t) * dt + S sqrt(dt) with S ~ N(0, Q) 
    X(t+dt/2) = (Y(t) + Y(t+dt))/2

    Args:
        L (_type_): _description_
        noise_cov (_type_): _description_
        x_0 (_type_, optional): _description_. Defaults to None.
        n_times (int, optional): _description_. Defaults to 100.
        dt (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    n_components = L.shape[0]
    if x_0 is None:
        x_0 = np.zeros(n_components)

    # Helper variable: Y(t+dt) = Y(t) + L @ Y(t) * dt + S sqrt(dt) with S ~ N(0, Q) 
    Y = [x_0]
    dt_int = dt*2
    for i in range(n_times):
        Y.append(Y[i-1] + L @ Y[i-1] * dt_int 
                 + np.random.multivariate_normal(np.zeros(n_components),
                                                 cov=noise_cov) * np.sqrt(dt_int))
    Y = np.array(Y)
    # X(t+dt/2) = (Y(t) + Y(t+dt))/2
    X = np.array([Y[:-1], Y[1:]]).mean(axis=0)

    times = np.arange(n_times) * dt
    X = np.array(X).T
    return times, X


def load_lim(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return

    print(f'Load LIM from file {filename}')
    with open(filename, 'rb') as f:
        store_dict = pickle.load(f)
    
    model = LIM(store_dict['tau'])
    model.G = store_dict['G']
    model.L = store_dict['L']
    model.Q = store_dict['Q']

    return model

# ======================================================================================
# Stationary LIM
# ======================================================================================

class LIM:
    """Create Linear Inverse model.

    Args:
        tau (int): Time-lag.
    """
    def __init__(self, tau) -> None:
        self.tau_0 = tau

        # Empty class variables
        self.G = None
        self.L = None
        self.Q = None
    
    def fit(self, data):
        """Fit LIM to data.

        Args:
            data (np.ndarray): Input data to estimate Green's function from.
                Dimensions (n_components, n_time). 
        """
        x = data[:, :-self.tau_0]
        x_tau = data[:, self.tau_0:]
        assert x.shape == x_tau.shape
        n_time = data.shape[1] - self.tau_0

        # Covariance matrices
        self.C_0 = (x @ x.T) / n_time
        self.C_tau = (x_tau @ x.T) / n_time

        # Compute inverse
        # np.linalg.inv
        # C_0_inv = np.linalg.inv(self.C_0) 
        # Cholesky decomposition
        L, low = linalg.cho_factor(self.C_0)
        C_0_inv = linalg.cho_solve((L, low), np.eye(self.C_0.shape[0]))
        # Verify: print(np.round(self.C_0 @ C_0_inv))
        
        # Compute time-evolution operator
        self.G = self.C_tau @ C_0_inv

        # Compute L = ln(G)/tau.
        # ======================
        # TODO: Decomposition of G could be stored
        w, U, V = matrix_decomposition(self.G)

        # Sort by decay time
        t_decay = - self.tau_0 / np.log(w)
        idx_sort = np.argsort(t_decay)[::-1]
        w = w[idx_sort]
        U = U[:,idx_sort]
        V = V[:,idx_sort]

        # Normalize such that u @ v.T = 1 and u.T @ v = 1
        # only required for G = U @ W @ V.T
        weights = U.T @ V
        U_norm = U @ np.linalg.inv(weights)
        
        log_W = np.diag(np.log(w) / self.tau_0) 
        # Compute L = U @ log(W)/tau @ U^-1
        L = U @ log_W @ np.linalg.inv(U)
        # Compute L = U @ log(W)/tau @ V.T 
        # L = U_norm @ log_W @ V.T
        
        # Nyquist mode check
        eps = 1e-5
        if np.max(np.abs(np.imag(L))) > eps:
            print("WARNING: Risk of nyquist mode.")
            print(f"WARNING: The imaginary part of L is {np.max(np.abs(np.imag(L)))}!")
            print(f"WARNING: Eigval of G are [{np.min(w)}, {np.max(w)}]!")
            self.L = L
        else:
            self.L = np.real(L)

        self.Q = self.noise_covariance()

        return None


    def noise_covariance(self):
        """Estimate noise covariance using the stationarity assumption.
            0 = L @ C_0 + C_0 @ L.T + Q 
        Returns:
            Q (np.ndarray): Estimated noise covariance of 
                dimensions (n_components, n_components).
        """
        Q = - self.L @ self.C_0 - self.C_0 @ self.L.T

        # Get eigenvalues and eigenvectors of Q
        w , U = np.linalg.eigh(Q)

        eps = 1e-5
        if np.max(np.abs(np.imag(w))) > eps:
            print(f"WARNING: Imaginary part of max eigval of Q are {np.max(np.abs(np.imag(w_Q)))}!")

        # Rescale Q if eigenvalues are negative
        if np.min(w) < 0:
            print(f"WARNING: Q has negative eigenvalues! Rescale Q!")

            trace_init = np.sum(w)
            w[w<0] = 0 + 1e-15
            trace_post = np.sum(w)
            scaling_factor = trace_init / trace_post
            w *= scaling_factor

            Q = U @ np.diag(w) @ U.T

        return Q
    

    def decaytime(self):
        """Decay time and period of eigenvalues.
        
        Returns:
            decay_time (np.ndarray): Decay time of eigenvalues of G.
            period (np.ndarray): Period of eigenvalues of G.
        
        """
        w, U, V = matrix_decomposition(self.G)

        decay_time = - self.tau_0 / np.real(np.log(w))
        period =  (2 * np.pi) * self.tau_0 / np.imag(np.log(w))

        return decay_time, period


    def forecast_mean(self, x, lag=1):
        """Forecasting mean. 

            x(t+tau) = G x(t)

        Args:
            x (np.ndarray): Input data to estimate Green's function from.
                Dimensions (n_components, n_time). 

        Returns:
            x_frcst (np.ndarray): Forecast.
                Dimensions (n_components, n_time). 
        """
        w, U, V = matrix_decomposition(self.G)

        # Sort by decay time
        t_decay = - self.tau_0 / np.log(w)
        idx_sort = np.argsort(t_decay)[::-1]
        w = w[idx_sort]
        U = U[:,idx_sort]
        V = V[:,idx_sort]

        weights = U.T @ V
        U_norm = U @ np.linalg.inv(weights)

        G_tau = U_norm @ np.diag(w**(lag/self.tau_0)) @ V.T 
#        G_tau = U @ np.diag(w**(lag/self.tau_0)) @ np.linalg.inv(U)
        x_frcst = np.einsum('ij,jk', np.real(G_tau), x)


        return x_frcst


    def forecast_use_L_mean(self, x, lag=1):
        """Forecasting data using the time-evolution operator L. 
        
            x(t+tau) = exp(L tau) x(t)

        Args:
            x (np.ndarray): Input data to estimate Green's function from.
                Dimensions (n_components, n_time). 

        Returns:
            x_frcst (np.ndarray): Forecast.
                Dimensions (n_components, n_time). 
        """
        n_components, n_times = x.shape
        x_frcst = np.zeros((n_components, n_times))
        for i in range(n_times):
            x_frcst[:, i] = np.dot(linalg.expm(self.L*lag), x[:, i])

        return x_frcst


    def euler_integration(self, x0: np.ndarray, dt:float, T: int, 
                          num_samples: int=1) -> tuple:
        """ Solve the SDE using the Euler method.

        Args:
            x0 (np.ndarray): Initial condition (num_components,)
            dt (float): Time step
            T (int): Total simulation time
            num_samples (int): Number of samples to generate
        Returns:
            times (np.ndarray): Time points of the simulation with 
                dim (num_steps+1)
            X (np.ndarray): Solution of the SDE with dim 
                (num_samples, num_steps+1, num_components)
        """
        num_steps = int(T/dt)
        num_components = len(x0)

        X = np.zeros((num_samples, num_steps+1, num_components))
        for n in range(num_samples):
            # Initialize solution
            X[n, 0, :] = x0

            # Iteratively apply the Euler method
            for i in range(num_steps):
                X[n, i+1, :] = X[n, i, :] + dt * np.dot(self.L, X[n, i, :])
                X[n, i+1, :] += np.random.multivariate_normal(np.zeros(num_components),
                                                           cov=self.Q) * np.sqrt(dt)
        times = np.arange(0, T+dt, dt)
        return times, X
    

    def rollout_mean(self, x_init: np.ndarray, lag_arr: np.ndarray) -> np.ndarray:
        """Rollout forecast for given lags.

        Args:
            x_init (np.ndarray): Initial values for prediction
                of shape (n_components, n_times)
            lag_arr (np.ndarray): List of lag times of shape (n_lags) 

        Returns:
            x_pred (np.ndarray): LIM predictions of shape (n_lags, n_components, n_times)
        """
        x_pred = []
        for lag in lag_arr:
            x_lag = np.real(self.forecast_mean(x_init, lag=lag))
            x_pred.append(x_lag)

        return np.array(x_pred)
 

    def error_covariance(self, power=1):
        """Error covariance.

        e**2 = C_0 - G_tau C_0 G_tau.T

        Args:
            power (int, optional): Power of G, i.e. G**power. For example power=tau/lag.
                Defaults to 1.

        Returns:
            error_cov (np.ndarray): Error covariance of dimension 
                (n_components, n_components). 
        """
        w, U, V = matrix_decomposition(self.G)
        G_tau = U @ np.diag(w**power) @ np.linalg.inv(U) 
        error_cov = self.C_0 - G_tau @ self.C_0 @ G_tau.T

        return error_cov
    

    def growth(self, lag=1):
        """Get growth of initial structure using SVD, i.e. largest eigenvalue of G.T @ G.

        Returns:
           w (float): Largest growth.  
           v (float): Inititial condition corresponding to growth.  
        """
        # Compute growth
        w, U, V = matrix_decomposition(self.G)
        G_tau = U @ np.diag(w**(lag/self.tau_0)) @ np.linalg.inv(U)

        # SVD of G_tau
        U_tau, w_tau, V_tau = np.linalg.svd(G_tau)
        growth = w_tau[0]**2
        init_condition=V_tau.T[:,0]

        return growth, init_condition 
    
    def growth_eigdec(self, lag=1, norm=None):
        """Get growth of initial structure using eigenvalue decomposition,

        Returns:
           w (float): Largest growth.  
           v (float): Inititial condition corresponding to growth.  
        """
        # Compute growth
        w, U, V = matrix_decomposition(self.G)
        G_tau = U @ np.diag(w**(lag/self.tau_0)) @ np.linalg.inv(U)

        if norm is not None:
            N = norm @ norm.T
            assert N.shape == G_tau.shape
            A = (G_tau.T @ N) @ G_tau
        else:
            A = G_tau.T @ G_tau

        # Eigenvalue decomposition
        w_tau, U_tau, _ = matrix_decomposition(G_tau.T @ G_tau)
        growth = w_tau[0]
        init_condition=U_tau[:,0]

        return growth, init_condition 
    
    

    def lagged_correlation(self, lag_times):
        """Lagged correlation matrix of LIM.

        Args:
            lag_times (np.ndarray): Array of lag times.

        Returns:
            C_lag (np.ndarray): Lagged covariance matrix, C_lag @ C_0, of 
                LIM
        """
        C_lag = []
        for lag in lag_times:
            # LIM autocorrelation
            w, U, V = matrix_decomposition(self.G)
            G_tau = U @ np.diag(w**(lag/self.tau_0)) @ np.linalg.inv(U)
            C_lag.append(G_tau @ self.C_0)
        
        return np.array(C_lag) 
    

    def save(self, filename, **kwargs):
        """Save LIM model to file.

        Args:
            filename (str): Filename. 
        """
        store_dict = dict(
            tau=self.tau_0,
            G=self.G,
            L=self.L,
            Q=self.Q
        )

        store_dict.update(**kwargs)
        
        with open(filename, 'wb') as f:
            pickle.dump(
                store_dict, f
            )

        print(f'LIM has been saved to {filename}.')
        return
        
        

# ======================================================================================
# Cyclo-stationary LIM
# ======================================================================================
class CSLIM():
    """Create cyclo-stationary Linear Inverse model.

    Args:
        tau (int): Time-lag.
    """
    def __init__(self, tau=1) -> None:
        self.tau_0 = tau

        # Class variables
        self.C_0 = None
        self.G = None
        self.L = None
        self.Q = None
    
    def fit(self, data: np.ndarray, start_month: int, average_window: int = 3):
        """Fit LIM to data.

        Args:
            data (np.ndarray): Input data to estimate Green's function from.
                Dimensions (n_components, n_time).
            start_month (int): Month of first datapoint. 
            average_window (int, optional): Size of filter for covariance matrix.
                Defaults to 3.
        """
        n_components, n_times = data.shape
        n_years = n_times/12

        x = data[:, :-self.tau_0]
        x_tau = data[:, self.tau_0:]

        # Covariance and lagged covariance
        C_0 = np.ones(shape=(12, n_components, n_components)) * np.nan
        C_tau = np.ones(shape=(12, n_components, n_components)) * np.nan
        for i in range(12):
            # in case the starting month is not Jan
            idx_month= (start_month + i - 1) % 12 
            x_month = x[:, i::12]
            x_tau_month = x_tau[:, i::12]

            assert x_month.shape == x_tau_month.shape
            C_0[idx_month, :, :] = (x_month @ x_month.T) / n_years
            C_tau[idx_month, :, :] = (x_tau_month @ x_month.T) / n_years

        # Running average of C_0 and C_tau
        if average_window > 1:
            C_0_mean = np.ones_like(C_0) * np.nan
            C_tau_mean = np.ones_like(C_tau) * np.nan
            for i in range(12):
                C_0_arr = []
                C_tau_arr = []
                for j in np.arange(i-average_window//2, i+average_window//2 + 1, 1):
                    idx = j % 12
                    C_0_arr.append(C_0[idx]) 
                    C_tau_arr.append(C_tau[idx]) 

                C_0_mean[i,:,:] = np.mean(C_0_arr, axis=0)
                C_tau_mean[i,:,:] = np.mean(C_tau_arr, axis=0)
            # Overwrite after averaging
            self.C_0 = C_0_mean
            self.C_tau = C_tau_mean
        else:
            self.C_0 = C_0
            self.C_tau = C_tau

        # Compute time-evolution operator
        self.G = np.ones(shape=(12, n_components, n_components)) * np.nan
        self.L = np.ones(shape=(12, n_components, n_components)) * np.nan
        for i in range(12):
            # Compute time-evolution operator
            self.G[i] = self.C_tau[i] @ np.linalg.inv(self.C_0[i]) 

            # Compute L = ln(G)/tau.
            w, U, V = matrix_decomposition(self.G[i])
            # Sort by decay time
            t_decay = - self.tau_0 / np.log(w)
            idx_sort = np.argsort(t_decay)[::-1]
            w = w[idx_sort]
            U = U[:,idx_sort]
            V = V[:,idx_sort]
            # Normalize such that u @ v.T = 1 and u.T @ v = 1
            # only required for G = U @ W @ V.T
            weights = U.T @ V
            U_norm = U @ np.linalg.inv(weights)
            # Compute L = U @ log(W)/tau @ V.T 
            log_W = np.diag(np.log(w) / self.tau_0) 
            self.L[i] = np.real(U_norm @ log_W @ V.T)

        # Check for nyquist mode of G_T = G12 ... G1
        # ==========================================
        G_arr = [self.G[i] for i in range(12)]
        G_T = functools.reduce(np.dot, G_arr[::-1])
        w, U, V = matrix_decomposition(G_T)

        # Compute L_T 
        t_decay = - self.tau_0 / np.log(w)
        idx_sort = np.argsort(t_decay)[::-1]
        w = w[idx_sort]
        U = U[:,idx_sort]
        V = V[:,idx_sort]
        weights = U.T @ V
        U_norm = U @ np.linalg.inv(weights)
        log_W = np.diag(np.log(w) / self.tau_0) 
        L_T = np.real(U_norm @ log_W @ V.T)

        eps = 1e-5
        if np.max(np.abs(np.imag(L_T))) > eps:
            print("WARNING: Risk of nyquist mode.")
            print(f"WARNING: The imaginary part of L is {np.max(np.abs(np.imag(L_T)))}!")
            print(f"WARNING: Eigval of G are [{np.min(w)}, {np.max(w)}]!")

        
        return None


    def noise_covariance(self):
        """Estimate noise covariance using the stationarity assumption.
            Q = (Q(j+i) - Q(j-1)) / (2*tau) - L @ C_0 - C_0 @ L.T 
        Returns:
            Q (np.ndarray): Estimated noise covariance of 
                dimensions (n_components, n_components).
        """
        self.Q = []
        n_months = len(self.L)
        for  i in range(n_months):
            L_month = self.L[i]
            C_0_month = self.C_0[i]

            idx = (i+1) % n_months
            dQ_month = (self.C_0[idx] - self.C_0[i-1]) / (2*self.tau_0)

            Q_month = dQ_month - L_month @ C_0_month - C_0_month @ L_month.T

            # Get eigenvalues and eigenvectors of Q
            w , U = np.linalg.eigh(Q_month)

            # Rescale Q if eigenvalues are negative
            if np.min(w) < 0:
                print(f"WARNING: Q of month={i+1} has negative eigenvalues! Rescale Q!")

                trace_init = np.sum(w)
                w[w<0] = 0 + 1e-15
                trace_post = np.sum(w)
                scaling_factor = trace_init / trace_post
                w *= scaling_factor

                Q_month = U @ np.diag(w) @ U.T

            self.Q.append(Q_month)

        return self.Q

    def time_evolution_operator(self, month: int, lag: int) -> np.ndarray:
        """ Time-evolution operator for given month and lag.

            G_j+tau (tau) = G_j+tau ... G_j+1  

        Args:
            month (int): Month of the initial data point. 
            lag (int): Lag time.

        Returns:
            x_frcst (np.ndarray): Forecast. Dimensions (n_components). 
        """
        idx_month = month - 1
        G_arr = []
        for i in range(idx_month, idx_month+lag):
            idx = i % 12
            G_arr.append(self.G[idx])

        G_tau = functools.reduce(np.dot, G_arr[::-1])

        return G_tau
    
    
    def error_covariance(self, init_month: int, lag: int):
        """Error covariance for month and lag time.
            
            e(tau, j) = C_{j+tau}(0) - G_j(tau) C_j (0) G_j(tau)^T

            j = init_month

        Args:
            init_month (int): Month of the initial data point.
            lag (int): Lag time.

        Returns:
            error_cov: Error covariance. 
        """

        # Error covariance e(tau, month) = C_0 - G_tau C_0 G_tau^T
        idx_init_month = init_month - 1
        idx_target_month = (init_month + lag -1) % 12

        C_0_init = self.C_0[idx_init_month]
        C_0_target = self.C_0[idx_target_month]

        G_tau = self.time_evolution_operator(init_month, lag)

        error_cov = C_0_target - G_tau @ C_0_init @ G_tau.T 

        return error_cov

    

    def forecast_mean(self, x: np.ndarray, month: int, lag: int) -> np.ndarray:
        """Forecasting mean. 

            x(t+tau) = G_(j+tau-1) ... G_(j) x_j(t)

        Args:
            x (np.ndarray): Initial data for forecast. Dimensions (n_components). 
            month (int): Month of the initial data point. 
            lag (int): Lag time.

        Returns:
            x_frcst (np.ndarray): Forecast. Dimensions (n_components). 
        """
        G_tau = self.time_evolution_operator(month, lag)
        x_frcst = G_tau @ x

        return x_frcst


    def rollout_mean(self, x_init: np.ndarray, init_month: int, 
                     lag_arr: np.ndarray) -> np.ndarray:
        """Rollout forecast for given lags.

        Args:
            x_init (np.ndarray): Initial values for prediction
                of shape (n_components)
            init_month (int): Month of the initial data point. 
            lag_arr (np.ndarray): List of lag times of shape (n_lags) 

        Returns:
            x_pred (np.ndarray): LIM predictions of shape (n_lags, n_components)
        """
        x_pred = []
        for lag in lag_arr:
            x_lag = np.real(self.forecast_mean(x_init, init_month, lag=lag))
            x_pred.append(x_lag)

        return np.array(x_pred)
    
    
    def euler_integration(self, x0: np.ndarray, month0: int,
                          dt:float, T: int, num_samples: int=1) -> tuple:
        """ Solve the SDE using the Euler method.

        Args:
            x0 (np.ndarray): Initial condition (num_components,)
            month0 (int): Month of initial condition
            dt (float): Time step
            T (int): Total simulation time
            num_samples (int): Number of samples to generate
        Returns:
            times (np.ndarray): Time points of the simulation with 
                dim (num_steps+1)
            X (np.ndarray): Solution of the SDE with dim 
                (num_samples, num_steps+1, num_components)
        """
        num_steps = int(T/dt)
        num_components = len(x0)
        times = np.arange(0, T+dt, dt)

        X = np.zeros((num_samples, num_steps+1, num_components))
        for n in range(num_samples):
            # Initialize solution
            X[n, 0, :] = x0
            idx_month = month0 - 1

            # Iteratively apply the Euler method
            for i, t in enumerate(times[:-1]):
                idx = int(idx_month + t) % 12
                L, Q = self.L[idx], self.Q[idx]

                X[n, i+1, :] = X[n, i, :] + dt * np.dot(L, X[n, i, :])
                X[n, i+1, :] += np.random.multivariate_normal(np.zeros(num_components),
                                                              cov=Q) * np.sqrt(dt)

        return times, X


    def growth(self, month: int = 1, lag:int = 1) -> tuple:
        """Get growth of initial structure using SVD.

        Args:
            month (int, optional): _description_. Defaults to 1.
            lag (int, optional): _description_. Defaults to 1.

        Returns:
           w (float): Largest growth.  
           v (np.ndarray): Inititial condition corresponding to growth.  
        """
        idx_month = month - 1
        G_arr = []
        for i in range(idx_month, idx_month+lag):
            idx = i % 12
            G_arr.append(self.G[idx])

        G_tau = functools.reduce(np.dot, G_arr[::-1])

        # SVD of G_tau
        _, w_tau, V_tau = np.linalg.svd(G_tau)
        growth = w_tau[0]**2
        init_condition=V_tau.T[:,0]

        return growth, init_condition 
