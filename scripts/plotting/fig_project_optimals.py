"""Figure 6: Predictability is captured by linear optimals.

(a) CS-LIM optimal initial condition (OIC) for a `tau`-month forecast in
    `init_month` (SSTA color, SSHA contours).
(b) The OIC evolved by `tau` months (El Niño-like pattern).
(c) ACC over lead time of the CS-LIM, LIM-LSTM and LSTM forecasts, stratified by
    the amplitude of the initial state's projection onto the OIC: forecasts from
    the weakest (0-10%, x) vs. strongest (90-100%, o) optimal initial growth.
"""

import cartopy as ctp
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import xarray as xr
from utils import base_parser
from utils import display_label
from utils import gpl
from utils import load_experiments
from utils import load_raw_data_and_eof
from utils import model_color
from utils import plt
from utils import preproc
from utils import save_figure

PERCENTILE_RANGES = {"weak": (0, 10), "strong": (90, 100)}
PERCENTILE_MARKER = {"weak": "x", "strong": "o"}
SSTA_ARGS = dict(vmin=-3, vmax=3, eps=0.5, cmap="RdBu_r", centercolor="white")
SSHA_ARGS = dict(
    kwargs_pl={"colors": "k", "levels": [-2.5, -1.25, 1.25, 2.5], "linewidths": 1.0},
    zerolinecolor=None,
    add_inline_labels=False,
)


def project_on_oic(z, optimal_init_pc, init_month, tau):
    """|Projection| of states initialized in `init_month` onto the tau-lead OIC."""
    idx = np.argwhere(z["time"].dt.month.values == init_month).flatten()[:-tau]
    z_init = z.isel(time=idx)
    opt = optimal_init_pc.sel(month=init_month, lag=tau)
    proj = xr.DataArray(opt.data @ z_init.data.T, coords=dict(time=z_init["time"].data))
    return np.abs(proj)


def percentile_acc(score, z_test, optimal_init_pc, var="ssta"):
    """Mean ACC per (lag, init_month, percentile group) for one model.

    For each lead and init month, group the initial states by the amplitude of
    their OIC projection and average the forecast pattern-ACC over each group.
    """
    months = optimal_init_pc["month"].data
    per_lag = []
    for tau in score["lag"].values:
        per_month = []
        for init_month in months:
            proj = project_on_oic(z_test, optimal_init_pc, init_month, int(tau))
            per_group = []
            for pmin, pmax in PERCENTILE_RANGES.values():
                lo, hi = np.percentile(proj, pmin), np.percentile(proj, pmax)
                sel = proj["time"].values[(proj.values >= lo) & (proj.values <= hi)]
                valid = preproc.add_to_cftime(sel, n_month=int(tau))
                valid = np.intersect1d(valid, score["time"].values)
                per_group.append(score[var].sel(time=valid, lag=tau).mean("time"))
            per_month.append(xr.concat(per_group, dim=pd.Index(list(PERCENTILE_RANGES), name="percentile")))
        per_lag.append(xr.concat(per_month, dim=pd.Index(months, name="init_month")))
    return xr.concat(per_lag, dim=pd.Index(score["lag"].data, name="lag"))


def main():
    parser = base_parser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["LIM", "LIM+LSTM", "LSTM"])
    parser.add_argument("--init_month", type=int, default=4, help="Init month for the maps.")
    parser.add_argument("--tau", type=int, default=12, help="Lead time for the maps [months].")
    parser.add_argument("--var", type=str, default="ssta", help="Variable scored in panel (c).")
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)
    lim_dir = experiments["LIM"]["paths"][0]

    print("Generating Figure 6 (predictability / optimal initial growth)...")
    raw = load_raw_data_and_eof()
    z_test = (
        raw["data_pc"][args.datasplit] if args.datasplit in raw["data_pc"] else raw["combined_eof"].transform(raw["ds"])
    )

    opt_dir = lim_dir / "metrics"
    optimal_init_pc = xr.open_dataarray(opt_dir / "optimal_init_pc.nc")
    init_map = xr.open_dataset(opt_dir / "optimal_init_map.nc")
    evolved_map = xr.open_dataset(opt_dir / "optimal_evolved_map.nc")
    # Normalize patterns to unit std and orient so El Niño growth is positive.
    init_map = -init_map / init_map.std()
    evolved_map = -evolved_map / evolved_map.std()

    # Percentile-stratified ACC per model.
    scores_perc = {}
    for model in args.models:
        cfg = experiments.get(model)
        score_file = cfg["paths"][0] / "metrics" / f"timescore_cc_{args.datasplit}.nc"
        if not score_file.exists():
            print(f"  [WARN] {model}: {score_file.name} not found, skipping")
            continue
        score = xr.open_dataset(score_file)
        scores_perc[model] = percentile_acc(score, z_test, optimal_init_pc, args.var)

    # --- Figure layout: maps (left), ACC (right) ---
    proj = ctp.crs.PlateCarree(central_longitude=180)
    fig = plt.figure(figsize=(7.5, 3.6))
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 4], height_ratios=[2, 2, 0.12], hspace=0.25, wspace=0.25)
    axs = []

    # (a) Optimal initial, (b) evolved optimal.
    im = None
    for row, mp in enumerate([init_map, evolved_map]):
        ax = fig.add_subplot(gs[row, 0], projection=proj)
        axs.append(ax)
        out = gpl.plot_map(
            mp["ssta"].sel(month=args.init_month, lag=args.tau),
            ax=ax,
            central_longitude=180,
            add_bar=False,
            **SSTA_ARGS,
        )
        im = out["im"]
        c = gpl.plot_contour(
            mp["ssha"].sel(month=args.init_month, lag=args.tau),
            ax=ax,
            central_longitude=180,
            **SSHA_ARGS,
            kwargs_labels=dict(fmt="%.2f"),
        )
        out["gl"].bottom_labels = row == 1
        c["gl"].bottom_labels = row == 1

    cax = fig.add_subplot(gs[2, 0])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="SSTA [norm.]")

    # (c) ACC vs lead time, per model and percentile group.
    ax = fig.add_subplot(gs[:, 1])
    axs.append(ax)
    for model, score in scores_perc.items():
        clr = model_color(experiments, model)
        for group in PERCENTILE_RANGES:
            score.sel(percentile=group).mean("init_month").plot(
                ax=ax, color=clr, marker=PERCENTILE_MARKER[group], markersize=4
            )
    ax.set_title("")
    ax.set_xlabel(r"$\tau$ [month]")
    ax.set_ylabel("ACC")
    ax.set_ylim(0.0, 1.1)
    ax.axhline(0.5, linestyle="--", color="gray")
    if scores_perc:
        ax.set_xticks(next(iter(scores_perc.values()))["lag"].values)

    # Legend: model colors + percentile markers.
    handles = [mlines.Line2D([], [], color=model_color(experiments, m), label=display_label(m)) for m in scores_perc]
    handles += [
        mlines.Line2D([], [], color="k", marker="o", ls="-", label=r"$\geq 90\,\%$"),
        mlines.Line2D([], [], color="k", marker="x", ls="-", label=r"$\leq 10\,\%$"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize="small")

    gpl.enumerate_axes(axs, pos_x=0.01, pos_y=0.98)
    save_figure(fig, f"fig6_project_optimals_month{args.init_month}.pdf", args.output)


if __name__ == "__main__":
    main()
