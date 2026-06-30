"""Figure 3: Spatial distribution of skill improvement.

Per-gridpoint RMSE skill score (vs. monthly climatology) at a fixed lead time:
 - left column  : absolute skill of the reference model (CS-LIM), SSTA / SSHA;
 - right column : skill difference (LIM-LSTM minus CS-LIM), SSTA / SSHA, where
                  red = the hybrid improves on the LIM.

Significance: the manuscript stipples the difference where a bootstrap t-test is
significant. That test needs per-gridpoint, per-time bootstrap skill, which is
NOT among the saved metrics (and the LIM-LSTM has no saved grid-space forecasts),
so it cannot be recomputed here. If a pvalue file is produced at the eval stage,
pass it via --significance to overlay the stippling.
"""

import cartopy as ctp
import numpy as np
import xarray as xr
from utils import base_parser
from utils import display_label
from utils import gpl
from utils import load_experiments
from utils import plt
from utils import save_figure

# Plot parameters per score: absolute skill (left col) and difference (right col).
ABS_PARAMS = {
    "mse": dict(cmap="plasma", vmin=0, vmax=1.5, eps=0.25),
    "rmsess": dict(cmap="plasma_r", vmin=-0.1, vmax=0.5, eps=0.1),
    "cc": dict(cmap="RdBu_r", vmin=-1, vmax=1, eps=0.1, centercolor="#FFFFFF"),
    "crpss": dict(cmap="viridis", vmin=0, vmax=0.5, eps=0.05),
}
DIFF_PARAMS = {
    "mse": dict(cmap="RdBu", vmin=-0.1, vmax=0.1, eps=0.02, centercolor="#FFFFFF"),
    "rmsess": dict(cmap="RdBu_r", vmin=-0.1, vmax=0.1, eps=0.02, centercolor="#FFFFFF"),
    "cc": dict(cmap="RdBu_r", vmin=-0.1, vmax=0.1, eps=0.02, centercolor="#FFFFFF"),
    "crpss": dict(cmap="RdBu_r", vmin=-0.1, vmax=0.1, eps=0.02, centercolor="#FFFFFF"),
}


def load_gridscore(experiments, model, scorekey, datasplit):
    cfg = experiments[model]
    path = cfg["paths"][0] / "metrics" / f"gridscore_{scorekey}_{datasplit}.nc"
    if not path.exists():
        raise FileNotFoundError(f"{model}: {path} not found")
    return xr.open_dataset(path)


def stipple(ax, mask):
    """Overlay stippling where `mask` (lat, lon) is True (significant)."""
    lon, lat = np.meshgrid(mask["lon"], mask["lat"])
    sel = mask.values
    ax.scatter(lon[sel], lat[sel], s=0.4, color="k", alpha=0.6, transform=ctp.crs.PlateCarree())


def main():
    parser = base_parser(description=__doc__)
    parser.add_argument("--reference", type=str, default="LIM", help="Reference model for the absolute-skill column.")
    parser.add_argument(
        "--compare", type=str, default="LIM+LSTM", help="Model compared against the reference (difference column)."
    )
    parser.add_argument("--scorekey", type=str, default="rmsess", choices=list(ABS_PARAMS.keys()))
    parser.add_argument("--lag", type=int, default=12, help="Lead time [months].")
    parser.add_argument("--vars", nargs="+", default=["ssta", "ssha"])
    parser.add_argument(
        "--significance",
        type=str,
        default=None,
        help="Optional netcdf with a boolean significance mask " "(vars, lat, lon) for the difference column.",
    )
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)
    scorekey, lag, variables = args.scorekey, args.lag, args.vars

    print("Generating Figure 3 (spatial skill maps)...")
    score_ref = load_gridscore(experiments, args.reference, scorekey, args.datasplit)
    score_cmp = load_gridscore(experiments, args.compare, scorekey, args.datasplit)

    signif = None
    if args.significance:
        from pathlib import Path

        if Path(args.significance).exists():
            signif = xr.open_dataset(args.significance)
        else:
            print(f"  [WARN] significance file not found: {args.significance}")

    proj = ctp.crs.PlateCarree(central_longitude=180)
    nrows = len(variables)
    fig, axs = plt.subplots(nrows, 2, figsize=(10, 2.6 * nrows), subplot_kw={"projection": proj})
    axs = np.atleast_2d(axs)

    im_abs = im_diff = None
    for i, var in enumerate(variables):
        # Left column: absolute skill of the reference model.
        im_abs = gpl.plot_map(
            score_ref[var].sel(lag=lag), ax=axs[i, 0], central_longitude=180, add_bar=False, **ABS_PARAMS[scorekey]
        )

        # Right column: difference (compare - reference); red = improvement.
        diff = score_cmp[var].sel(lag=lag) - score_ref[var].sel(lag=lag)
        im_diff = gpl.plot_map(diff, ax=axs[i, 1], central_longitude=180, add_bar=False, **DIFF_PARAMS[scorekey])

        if signif is not None and var in signif:
            stipple(axs[i, 1], signif[var].astype(bool))

        axs[i, 0].text(
            -0.12, 0.5, var.upper(), va="center", ha="center", rotation="vertical", transform=axs[i, 0].transAxes
        )

    ref_lbl = display_label(args.reference)
    cmp_lbl = display_label(args.compare)
    axs[0, 0].set_title(ref_lbl)
    axs[0, 1].set_title(rf"{cmp_lbl} $-$ {ref_lbl}")

    score_name = scorekey.upper()
    fig.colorbar(
        im_abs["im"],
        ax=axs[:, 0].tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        extend="both",
        label=score_name,
    )
    fig.colorbar(
        im_diff["im"],
        ax=axs[:, 1].tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.08,
        extend="both",
        label=rf"$\Delta$ {score_name}",
    )

    gpl.enumerate_axes(axs, pos_x=0.01, pos_y=0.95, fontsize="medium")
    save_figure(fig, f"fig3_skill_map_{scorekey}_lag{lag}.pdf", args.output)


if __name__ == "__main__":
    main()
