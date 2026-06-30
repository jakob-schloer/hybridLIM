"""Shared utilities for the paper figure scripts.

Single source of truth for model paths/colors/training-data sweeps is
``experiments.yaml`` (see CLAUDE.md). All figure scripts load it through
``load_experiments`` and resolve scores through the helpers below, so paths and
colors are never duplicated across scripts.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import OmegaConf

# Add project root to path so `hyblim` is importable when run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from hyblim.data import eof  # noqa: E402
from hyblim.data import preproc  # noqa: E402
from hyblim.utils import metric  # noqa: E402

plt.style.use(str(PROJECT_ROOT / "paper.mplstyle"))

# Default locations
EXPERIMENTS_YAML = Path(__file__).resolve().parent / "experiments.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots/paper_figures"

# Training-data sweep (months); see CLAUDE.md "num_traindata sweeps".
NUM_DATA = [600, 1200, 2400, 3600, 6000, 9000, 12000, 18000]

# Raw CESM2 fields, needed by figures that reconstruct grid-space maps.
CESM2_PATHS = {
    "ssta": "data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssta_lat-31_33_lon130_290_gr1.0.nc",
    "ssha": "data/cesm2-picontrol/b.e21.B1850.f09_g17.CMIP6-piControl.001.pop.h.ssha_lat-31_33_lon130_290_gr1.0.nc",
}

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------


def load_experiments(yaml_path=EXPERIMENTS_YAML):
    """Load the experiment registry, resolving every path to an absolute Path.

    Relative paths in the yaml (e.g. ``${modelpath}/lim/...``) are interpreted
    relative to the yaml file's directory.
    """
    yaml_path = Path(yaml_path).resolve()
    config = OmegaConf.load(str(yaml_path))
    experiments = OmegaConf.to_container(config["experiments"], resolve=True)
    yaml_dir = yaml_path.parent
    for cfg in experiments.values():
        cfg["paths"] = [_resolve_path(p, yaml_dir) for p in cfg["paths"]]
    return experiments


def _resolve_path(p, base):
    p = Path(p)
    return p if p.is_absolute() else (base / p).resolve()


def experiment_name(model, num_data=None):
    """Registry key for a model, optionally for a training-data subset."""
    return model if num_data is None else f"{model}_n{num_data}"


# Registry keys -> manuscript legend labels (used by the skill-score figures).
DISPLAY_LABELS = {"LIM": "CS-LIM", "LIM+LSTM": "LIM-LSTM"}


def display_label(model, overrides=None):
    """Legend label for a model, with optional per-figure overrides."""
    if overrides and model in overrides:
        return overrides[model]
    return DISPLAY_LABELS.get(model, model)


def model_color(experiments, model):
    """Color of a (base) model from the registry, falling back to None."""
    cfg = experiments.get(model)
    return cfg.get("color") if cfg else None


# ---------------------------------------------------------------------------
# Nino-index forecast scores
# ---------------------------------------------------------------------------


def _nino_files(model_dir, datasplit):
    """Locate (forecast, target) nino files, trying metrics/ then the root."""
    for base in (model_dir / "metrics", model_dir):
        frcst = base / f"nino_frcst_{datasplit}.nc"
        target = base / f"nino_target_{datasplit}.nc"
        if frcst.exists() and target.exists():
            return frcst, target
    return None, None


def _load_nino_score(model_dir, datasplit):
    """Load a single experiment's nino scores; (None, None) if files missing."""
    frcst_file, target_file = _nino_files(model_dir, datasplit)
    if frcst_file is None:
        return None, None
    frcst = xr.open_dataset(frcst_file)
    target = xr.open_dataset(target_file).transpose("time", "lag")
    if "member" in frcst.dims:
        frcst = frcst.transpose("time", "member", "lag")
    return metric.time_series_score(frcst, target)


def load_nino_scores(experiments, models, datasplit="test"):
    """Nino scores for each model in `models`. Missing experiments are skipped.

    Returns (scores, scores_month), each a dict {model: {scorekey: DataArray}}.
    """
    scores, scores_month = {}, {}
    for model in models:
        cfg = experiments.get(model)
        if cfg is None:
            print(f"  [WARN] {model}: not in registry, skipping")
            continue
        s, sm = _load_nino_score(cfg["paths"][0], datasplit)
        if s is None:
            print(f"  [WARN] {model}: nino files for '{datasplit}' not found, skipping")
            continue
        scores[model], scores_month[model] = s, sm
    return scores, scores_month


def load_nino_scores_ndata(experiments, models, num_data=NUM_DATA, datasplit="test"):
    """Nino scores across the training-data sweep.

    Returns (scores, scores_month), each {model: {scorekey: DataArray}} where
    every DataArray has an extra leading ``ndata`` dimension. Models/subsets
    with no metrics on disk are silently skipped.
    """
    scores, scores_month = {}, {}
    for model in models:
        per_n, per_n_month, found = [], [], []
        for n in num_data:
            cfg = experiments.get(experiment_name(model, n))
            if cfg is None:
                continue
            s, sm = _load_nino_score(cfg["paths"][0], datasplit)
            if s is None:
                continue
            per_n.append(s)
            per_n_month.append(sm)
            found.append(n)
        if not found:
            print(f"  [WARN] {model}: no training-data subsets found, skipping")
            continue
        per_n = metric.listofdicts_to_dictoflists(per_n)
        per_n_month = metric.listofdicts_to_dictoflists(per_n_month)
        ndata_idx = pd.Index(found, name="ndata")
        scores[model] = {k: xr.concat(v, dim=ndata_idx) for k, v in per_n.items()}
        scores_month[model] = {k: xr.concat(v, dim=ndata_idx) for k, v in per_n_month.items()}
    return scores, scores_month


# ---------------------------------------------------------------------------
# Raw data + EOF (for grid-space reconstruction figures)
# ---------------------------------------------------------------------------

_cached_raw_data = {}


def load_raw_data_and_eof():
    """Load raw CESM2 data, normalizers, and EOFs (cached on first call)."""
    if _cached_raw_data:
        return _cached_raw_data

    print("  Loading raw CESM2 data + computing EOFs (this may take a minute)...")
    da_arr, normalizers = [], {}
    for var, rel_path in CESM2_PATHS.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            print(f"  [WARN] Raw data not found: {path}")
            return None
        da = xr.open_dataset(path)[var]
        norm = preproc.Normalizer()
        da = norm.fit_transform(da)
        da.attrs = norm.to_dict()
        da_arr.append(da)
        normalizers[var] = norm

    ds = xr.merge(da_arr)
    lsm = xr.open_dataset(PROJECT_ROOT / "data/land_sea_mask_common.nc")["lsm"]
    ds = ds.where(lsm != 1, other=np.nan)

    # EOFs: load the precomputed decomposition the hindcasts were built with, so
    # grid-space reconstructions are sign/order-consistent with them (a fresh fit
    # could flip EOF signs and corrupt the reconstructed maps).
    n_eof = [20, 10]
    eof_path = PROJECT_ROOT / "data/cesm2-picontrol/pca" / eof.eof_filename(list(ds.data_vars), n_eof)
    combined_eof = eof.get_combined_eof(
        ds,
        n_eof,
        eof_path=str(eof_path),
        train_period=(0, int(0.8 * len(ds["time"]))),
    )

    n_time = len(ds["time"])
    test_start = int(0.9 * n_time)
    data_pc = {"test": combined_eof.transform(ds.isel(time=slice(test_start, None)))}

    _cached_raw_data.update(
        dict(
            ds=ds,
            normalizers=normalizers,
            combined_eof=combined_eof,
            data_pc=data_pc,
        )
    )
    return _cached_raw_data


# ---------------------------------------------------------------------------
# Argument parsing & figure output
# ---------------------------------------------------------------------------


def base_parser(description):
    """Argument parser with the arguments common to all figure scripts."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--experiments", type=str, default=str(EXPERIMENTS_YAML), help="Path to the experiments registry yaml."
    )
    p.add_argument(
        "--datasplit",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on (default: test).",
    )
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output folder for the figure.")
    return p


def save_figure(fig, name, output_dir=DEFAULT_OUTPUT_DIR, dpi=300):
    """Save `fig` to `output_dir/name`, creating the folder if needed.

    Figures containing cartopy maps are saved as PNG (dense pcolormesh maps
    render poorly as vector PDF), the rest keep their requested extension
    (typically PDF). paper.mplstyle's ``figure.autolayout`` (tight_layout) is
    also disabled for map figures, as it is incompatible with GeoAxes.
    """
    has_maps = False
    try:
        from cartopy.mpl.geoaxes import GeoAxes

        has_maps = any(isinstance(ax, GeoAxes) for ax in fig.axes)
    except ImportError:
        pass

    if has_maps:
        fig.set_layout_engine("none")
        name = str(Path(name).with_suffix(".png"))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Saved {path}")
    return path
