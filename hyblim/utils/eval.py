"""Functions for evaluating models."""

import numpy as np
import pandas as pd
import torch
import xarray as xr

from hyblim.data import eof
from hyblim.data import preproc
from hyblim.utils import enso
from hyblim.utils import metric

# Niño region monitored during training (Niño4 is the paper's headline index).
HEADLINE_NINO = ["nino4"]


def latent_frcst_to_grid(z_hindcast: np.ndarray, times: np.ndarray, combined_eof, normalizers, extended_eof=None):
    """Transform forecast in latent space to grid space.

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
        x_var = np.einsum("ijk,kl->ijl", z_var, components)

        # Add nans and reshape to grid space
        n_times, n_members, _ = x_var.shape
        mask_map = eofa.ids_notNaN.unstack().drop_vars(["time", "month"])
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
            data=x_flat.reshape(n_times, n_members, *mask_map.shape),
            coords=dict(time=times, member=np.arange(1, n_members + 1), **mask_map.coords),
            name=var,
        )
        # Invert normalization
        x_map = normalizers[var].inverse_transform(x_map)
        x_hindcast.append(x_map)
        n_start = n_end

    return xr.merge(x_hindcast)


def ensemble_grid_to_indices(x_grid: xr.Dataset) -> xr.Dataset:
    """Reduce a grid-space forecast/target to the monitoring indices.

    Returns the headline Niño indices (from SSTA) plus the domain-mean of every
    variable (``<var>_mean``). Leading dimensions (e.g. time, member, lag) are kept,
    spatial dimensions are reduced. Input must be in physical (un-normalized) units.

    Args:
        x_grid (xr.Dataset): Grid-space data with 'lat'/'lon' dims and an 'ssta' variable.

    Returns:
        xr.Dataset: Index time series with the spatial dims reduced.
    """
    nino = enso.get_nino_indices(x_grid["ssta"])
    data = {name: nino[name] for name in HEADLINE_NINO}
    for var in x_grid.data_vars:
        data[f"{var}_mean"] = x_grid[var].mean(dim=("lat", "lon"), skipna=True)
    return xr.Dataset(data)


def _index_array_to_ds(arr: np.ndarray, dims: list, index_names: list, lag_arr: list) -> xr.Dataset:
    """Wrap a small index array (..., n_index) into a Dataset for `index_verification_metrics`."""
    n_time = arr.shape[0]
    coords = {"lag": list(lag_arr), "time": np.arange(n_time)}
    return xr.Dataset({name: (dims, arr[..., i]) for i, name in enumerate(index_names)}, coords=coords)


class LatentIndexMonitor:
    """Fast per-lead Niño4 + field-mean validation metrics for PC-space ensembles.

    The Niño4 box mean and the domain means are *linear* functionals of the PCs, so the
    "reconstruct grid -> average region" step collapses into a single precomputed matrix
    (``index = z_norm @ W + b``) applied with one torch matmul per epoch. The full-field
    target indices do not change between epochs and are precomputed once.

    Index order: Niño4 (from SSTA), then the domain mean of each variable (``<var>_mean``),
    matching `ensemble_grid_to_indices`.
    """

    # (index name, source variable, region) — region is 'nino4' (box) or 'domain' (full field).
    INDEX_SPECS = [("nino4", "ssta", "nino4"), ("ssta_mean", "ssta", "domain"), ("ssha_mean", "ssha", "domain")]

    def __init__(self, combined_eof, normalizer_pca, ds, val_times, lag_arr, n_members=16):
        self.lag_arr = list(lag_arr)
        self.n_members = n_members
        self.index_names = [name for name, _, _ in self.INDEX_SPECS]

        pc_std = np.asarray(normalizer_pca.std.values, dtype=np.float64)
        pc_mean = np.asarray(normalizer_pca.mean.values, dtype=np.float64)
        grid_norm = {v: preproc.normalizer_from_dict(ds[v].attrs) for v in ds.data_vars}

        # eof block (start, end) per variable
        blocks, start = {}, 0
        for v, eofa in zip(combined_eof.vars, combined_eof.eofa_lst):
            blocks[v] = (start, start + eofa.n_components)
            start += eofa.n_components
        n_total = start

        W = np.zeros((n_total, len(self.INDEX_SPECS)))
        b = np.zeros(len(self.INDEX_SPECS))
        for j, (name, var, region) in enumerate(self.INDEX_SPECS):
            eofa = combined_eof.eofa_lst[combined_eof.vars.index(var)]
            w = self._region_weight(eofa, region)  # (n_comp_var,)
            s, e = blocks[var]
            gstd, gmean = float(grid_norm[var].std), float(grid_norm[var].mean)
            # index = gstd * (z_norm[block] * pc_std + pc_mean) @ w + gmean
            W[s:e, j] = gstd * pc_std[s:e] * w
            b[j] = gstd * float(pc_mean[s:e] @ w) + gmean
        self.W = torch.from_numpy(W).float()
        self.b = torch.from_numpy(b).float()

        # Full-field target indices over the validation times (constant across epochs).
        ds_val = ds.sel(time=val_times)
        ds_phys = xr.merge([grid_norm[v].inverse_transform(ds_val[v]) for v in ds_val.data_vars])
        target_ds = ensemble_grid_to_indices(ds_phys)
        self.target_arr = np.stack([target_ds[name].values for name in self.index_names], axis=-1)

    @staticmethod
    def _region_weight(eofa, region):
        """Mean of each EOF component over the region's ocean grid points (the box weights)."""
        comp_maps = xr.concat(
            [preproc.flattened2map(eofa.pca.components_[i], eofa.ids_notNaN) for i in range(eofa.n_components)],
            dim=pd.Index(np.arange(eofa.n_components), name="eof"),
        )
        if region == "domain":
            return comp_maps.mean(dim=("lat", "lon"), skipna=True).values
        elif region == "nino4":
            comp_maps.name = "ssta"
            return enso.get_nino_indices(comp_maps)["nino4"].values
        raise ValueError(f"Unknown region {region}")

    def compute(self, z_frcst: torch.Tensor, time_idx: np.ndarray, prefix: str = "val") -> dict:
        """Per-lead metrics from a PC-space ensemble forecast.

        Args:
            z_frcst (torch.Tensor): Normalized PC forecast (n, member, n_lag, n_components).
            time_idx (np.ndarray): Valid-time indices into `val_times` (n, >= max(lag_arr)).
            prefix (str): wandb key prefix.
        Returns:
            dict[str, float]: Flat metric dict (see `index_metrics_to_logdict`).
        """
        z = z_frcst.detach().to(torch.float32).cpu()
        idx_all = torch.einsum("nmlc,ci->nmli", z, self.W) + self.b  # (n, member, n_lag, n_index)
        cols = [lag - 1 for lag in self.lag_arr]
        frcst = idx_all[:, :, cols, :].numpy()
        target = self.target_arr[time_idx[:, cols]]
        frcst_ds = _index_array_to_ds(frcst, ["time", "member", "lag"], self.index_names, self.lag_arr)
        target_ds = _index_array_to_ds(target, ["time", "lag"], self.index_names, self.lag_arr)
        metrics = metric.index_verification_metrics(frcst_ds, target_ds, n_members=self.n_members)
        return index_metrics_to_logdict(metrics, self.lag_arr, prefix=prefix)


class GridIndexMonitor:
    """Fast per-lead Niño4 + field-mean validation metrics for grid-space ensembles.

    Region means are masked averages computed directly on the model grid output with
    torch (no per-batch xarray / EOF reconstruction). Index order matches
    `LatentIndexMonitor`: Niño4 (from SSTA), then ``<var>_mean`` per variable.
    """

    def __init__(self, ds, lag_arr, n_members=16):
        self.vars = list(ds.data_vars)
        self.index_names = ["nino4"] + [f"{v}_mean" for v in self.vars]
        self.lag_arr = list(lag_arr)
        self.n_members = n_members
        self.grid_norm = {v: preproc.normalizer_from_dict(ds[v].attrs) for v in self.vars}

        ocean = ds[self.vars[0]].isel(time=0).notnull()  # (lat, lon) bool
        lat, lon = ds["lat"], ds["lon"]
        nino4 = ocean & (lat >= -5) & (lat <= 5) & (lon >= 160) & (lon <= 210)  # cut_map box for Niño4
        self._ocean = torch.from_numpy(ocean.values).float()
        self._nino4 = torch.from_numpy(nino4.transpose("lat", "lon").values).float()

    @staticmethod
    def _masked_mean(field, mask):
        return (field * mask).sum(dim=(-2, -1)) / mask.sum()

    def reduce(self, tensor: torch.Tensor, has_member: bool) -> torch.Tensor:
        """Reduce a grid-space tensor to the monitoring indices.

        Args:
            tensor (torch.Tensor): (B, member, vars, lag, lat, lon) if `has_member` else
                (B, vars, lag, lat, lon), in normalized units.
        Returns:
            torch.Tensor: Index tensor (B, [member], lag, n_index).
        """
        var_axis = 2 if has_member else 1
        ocean = self._ocean.to(tensor.device)
        nino4 = self._nino4.to(tensor.device)
        ssta = tensor.select(var_axis, 0).float() * float(self.grid_norm[self.vars[0]].std) + float(
            self.grid_norm[self.vars[0]].mean
        )
        out = [self._masked_mean(ssta, nino4)]
        for vi, v in enumerate(self.vars):
            f = tensor.select(var_axis, vi).float() * float(self.grid_norm[v].std) + float(self.grid_norm[v].mean)
            out.append(self._masked_mean(f, ocean))
        return torch.stack(out, dim=-1)

    def metrics_logdict(self, frcst_idx: torch.Tensor, target_idx: torch.Tensor, prefix: str = "val") -> dict:
        """Per-lead metrics from accumulated index tensors (n, [member], lag, n_index)."""
        cols = [lag - 1 for lag in self.lag_arr]
        frcst = frcst_idx[:, :, cols, :].detach().cpu().numpy()
        target = target_idx[:, cols, :].detach().cpu().numpy()
        frcst_ds = _index_array_to_ds(frcst, ["time", "member", "lag"], self.index_names, self.lag_arr)
        target_ds = _index_array_to_ds(target, ["time", "lag"], self.index_names, self.lag_arr)
        metrics = metric.index_verification_metrics(frcst_ds, target_ds, n_members=self.n_members)
        return index_metrics_to_logdict(metrics, self.lag_arr, prefix=prefix)


def index_metrics_to_logdict(metrics: dict, lag_arr: list, prefix: str = "val") -> dict:
    """Flatten per-lead index metrics into a flat scalar dict for wandb logging.

    Emits one key per lead time (``<prefix>/<index>/<metric>/lag<n>``) and a lead-time
    average over `lag_arr` (``<prefix>/<index>/<metric>/mean``).

    Args:
        metrics (dict): Output of `metric.index_verification_metrics`.
        lag_arr (list): Lead times to log.
        prefix (str, optional): Key prefix. Defaults to "val".

    Returns:
        dict[str, float]: Flat scalar dict.
    """
    log = {}
    for metric_name, ds in metrics.items():
        for var in ds.data_vars:
            for lag in lag_arr:
                log[f"{prefix}/{var}/{metric_name}/lag{int(lag)}"] = float(ds[var].sel(lag=lag))
            log[f"{prefix}/{var}/{metric_name}/mean"] = float(ds[var].sel(lag=lag_arr).mean())
    return log


def latent_evaluation(
    z_hindcast: np.ndarray,
    time_idx: np.ndarray,
    times: np.ndarray,
    combined_eof: eof.CombinedEOF,
    ds_target: xr.Dataset,
    lag_arr: list = [1, 3, 6, 9, 12, 15, 18, 24],
    extended_eof: eof.CombinedEOF = None,
):
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
            z_hindcast[:, :, lag - 1, :],
            times[time_idx[:, lag - 1]],
            combined_eof,
            normalizers,
            extended_eof=extended_eof,
        )

        # Get target data
        x_target = ds_target.sel(time=times[time_idx[:, lag - 1]])
        # Unnormalize in data space
        x_target = xr.merge([normalizers[var].inverse_transform(x_target[var]) for var in x_target.data_vars])

        print("Compute metrics!", flush=True)
        x_frcst_mean = x_hindcast.mean(dim="member")
        x_frcst_std = x_hindcast.std(dim="member", ddof=1)
        n_members = len(x_hindcast["member"])

        # Compute metrics per gridpoint
        grid_verif = metric.verification_metrics_per_gridpoint(x_target, x_frcst_mean, x_frcst_std, n_members=n_members)
        grid_verif["lag"] = lag
        verification_per_gridpoint.append(grid_verif)

        # Compute metrics per time
        time_verif = metric.verification_metrics_per_time(x_target, x_frcst_mean, x_frcst_std, n_members=n_members)
        time_verif["lag"] = lag
        verification_per_time.append(time_verif)

        # Nino indices
        nino_index = {
            "target": enso.get_nino_indices(x_target["ssta"]),
            "frcst": enso.get_nino_indices(x_hindcast["ssta"]),
            "lag": lag,
        }
        nino_indices.append(nino_index)

    grid_scores = metric.listofdicts_to_dictofxr(verification_per_gridpoint, dim_key="lag")
    time_scores = metric.listofdicts_to_dictofxr(verification_per_time, dim_key="lag")
    nino_ids = metric.listofdicts_to_dictofxr(nino_indices, dim_key="lag")

    return grid_scores, time_scores, nino_ids
