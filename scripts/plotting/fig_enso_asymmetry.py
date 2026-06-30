"""Figure 7: Nonlinear models capture ENSO asymmetries.

For states initialized in `init_month` with the absolute largest optimal initial
growth (top decile of |projection onto the tau-lead OIC|), the tau-month forecasts
fall into warm (positive growth) and cold (negative growth) groups. We compose:

 - W - C (left column)  : warm minus cold -> the (symmetric) ENSO magnitude.
 - W + C (right column) : warm plus cold  -> the warm/cold asymmetry.

Rows are Target, CS-LIM, LIM-LSTM and LSTM. SSTA is shaded (masked where the
warm/cold difference is not significant at 95%), SSHA is overlaid as contours.

The warm/cold membership is determined once from the OIC projection (so every
model is composited over the same events); their orientation (which sign is
"warm") is fixed by the sign of the target's evolved Niño4.
"""

import cartopy as ctp
import numpy as np
import xarray as xr
from utils import base_parser
from utils import load_experiments
from utils import load_raw_data_and_eof
from utils import plt
from utils import preproc
from utils import save_figure

import hyblim.geoplot as gpl
from hyblim.utils import enso
from hyblim.utils import stats

# Model -> hindcast filename (PC-space forecasts, keyed by initialization time).
HINDCAST_FILE = {
    "LIM": "cslim_hindcast_ssta-ssha_eof20-10_{split}.nc",
    "LIM+LSTM": "limlstm_hindcast_ssta-ssha_eof20-10_{split}.nc",
    "LSTM": "lstm_hindcast_ssta-ssha_eof20-10_{split}.nc",
}
ROW_MODELS = ["Target", "LIM", "LIM+LSTM", "LSTM"]
ROW_LABELS = {"Target": "Target", "LIM": "CS-LIM", "LIM+LSTM": "LIM-LSTM", "LSTM": "LSTM"}

SSTA_ARGS = dict(cmap="RdBu_r", vmin=-3, vmax=3, eps=0.5, centercolor="white")
SSHA_ARGS = dict(
    kwargs_pl={"colors": "k", "levels": [-1.5, -0.75, 0.75, 1.5], "linewidths": 1.0},
    zerolinecolor=None,
    add_inline_labels=False,
)


def strong_growth_split(z_test, optimal_init_pc, init_month, tau, threshold):
    """Init times with top-`threshold`% |OIC projection|, split by projection sign."""
    idx = np.argwhere(z_test["time"].dt.month.values == init_month).flatten()[:-tau]
    z_init = z_test.isel(time=idx)
    opt = optimal_init_pc.sel(month=init_month, lag=tau)
    proj = xr.DataArray(opt.data @ z_init.data.T, coords=dict(time=z_init["time"].data))
    strong = np.abs(proj) >= np.percentile(np.abs(proj), threshold)
    pos = proj["time"].values[strong.values & (proj.values > 0)]
    neg = proj["time"].values[strong.values & (proj.values < 0)]
    return pos, neg


def forecast_event_fields(hindcast, init_times, lag, combined_eof, normalizer):
    """Per-event ensemble-mean forecast fields (SSTA in K, SSHA normalized)."""
    init_times = np.intersect1d(init_times, hindcast["time"].values)
    z = hindcast["z"].sel(time=init_times, lag=lag).mean("member")  # (event, eof)
    rec = combined_eof.reconstruction(z.values, times=np.arange(z.shape[0]))
    rec["ssta"] = normalizer.inverse_transform(rec["ssta"])
    return rec  # dim 'time' indexes the events


def target_event_fields(ds, init_times, tau, normalizer):
    """Per-event evolved target fields (SSTA in K, SSHA normalized)."""
    valid = preproc.add_to_cftime(list(init_times), n_month=tau)
    valid = np.intersect1d(valid, ds["time"].values)
    field = ds.sel(time=valid)
    return field.assign(ssta=normalizer.inverse_transform(field["ssta"]))


def composites(warm, cold):
    """(W-C, W+C) means and the significance masks of each (over the event dim)."""
    wmc, wpc, mask_mc, mask_pc = {}, {}, {}, {}
    for var in ["ssta", "ssha"]:
        wmc[var] = warm[var].mean("time") - cold[var].mean("time")
        wpc[var] = warm[var].mean("time") + cold[var].mean("time")
    # Significance on the shaded SSTA only: W-C tests warm != cold; W+C tests
    # warm != -cold (i.e. a non-zero warm/cold asymmetry).
    _, p_mc = stats.ttest_field(warm["ssta"], cold["ssta"])
    _, p_pc = stats.ttest_field(warm["ssta"], -cold["ssta"])
    mask_mc = stats.field_significance_mask(p_mc, corr_type=None)
    mask_pc = stats.field_significance_mask(p_pc, corr_type=None)
    return wmc, wpc, mask_mc, mask_pc


def main():
    parser = base_parser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["LIM", "LIM+LSTM", "LSTM"])
    parser.add_argument("--init_month", type=int, default=4, help="Init month for composites.")
    parser.add_argument("--tau", type=int, default=12, help="Lead time [months].")
    parser.add_argument(
        "--threshold", type=float, default=90, help="Percentile of |OIC projection| selecting strong-growth states."
    )
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)
    lim_dir = experiments["LIM"]["paths"][0]
    init_month, tau = args.init_month, args.tau

    print("Generating Figure 7 (ENSO asymmetry composites)...")
    raw = load_raw_data_and_eof()
    combined_eof, normalizer = raw["combined_eof"], raw["normalizers"]["ssta"]
    z_test = raw["data_pc"][args.datasplit]
    ds = raw["ds"]

    optimal_init_pc = xr.open_dataarray(lim_dir / "metrics" / "optimal_init_pc.nc")
    pos, neg = strong_growth_split(z_test, optimal_init_pc, init_month, tau, args.threshold)

    # Orient warm/cold by the sign of the target's evolved Niño4.
    tgt_pos = target_event_fields(ds, pos, tau, normalizer)
    tgt_neg = target_event_fields(ds, neg, tau, normalizer)
    n4_pos = float(enso.get_nino_indices(tgt_pos["ssta"].mean("time"))["nino4"])
    n4_neg = float(enso.get_nino_indices(tgt_neg["ssta"].mean("time"))["nino4"])
    if n4_pos >= n4_neg:
        warm_t, cold_t, warm_tgt, cold_tgt = pos, neg, tgt_pos, tgt_neg
    else:
        warm_t, cold_t, warm_tgt, cold_tgt = neg, pos, tgt_neg, tgt_pos
    print(f"  warm: {len(warm_t)} events, cold: {len(cold_t)} events")

    # Per-model warm/cold event fields (Target first, then the forecast models).
    fields = {"Target": (warm_tgt, cold_tgt)}
    for model in args.models:
        cfg = experiments.get(model)
        hc_path = cfg["paths"][0] / HINDCAST_FILE[model].format(split=args.datasplit)
        if not hc_path.exists():
            print(f"  [WARN] {model}: {hc_path.name} not found, skipping")
            continue
        hc = xr.open_dataset(hc_path)
        warm = forecast_event_fields(hc, warm_t, tau, combined_eof, normalizer)
        cold = forecast_event_fields(hc, cold_t, tau, combined_eof, normalizer)
        fields[model] = (warm, cold)

    rows = [m for m in ROW_MODELS if m in fields]

    # --- Figure: rows = models, cols = (W-C, W+C) ---
    proj = ctp.crs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(len(rows), 2, figsize=(9, 2.4 * len(rows)), subplot_kw={"projection": proj})
    axs = np.atleast_2d(axs)

    im = None
    for i, model in enumerate(rows):
        warm, cold = fields[model]
        wmc, wpc, mask_mc, mask_pc = composites(warm, cold)
        for j, (comp, mask) in enumerate([(wmc, mask_mc), (wpc, mask_pc)]):
            out = gpl.plot_map(
                comp["ssta"].where(mask), ax=axs[i, j], central_longitude=180, add_bar=False, **SSTA_ARGS
            )
            im = out["im"]
            gpl.plot_contour(comp["ssha"], ax=axs[i, j], central_longitude=180, **SSHA_ARGS)
            out["gl"].bottom_labels = i == len(rows) - 1
        axs[i, 0].text(
            -0.28,
            0.5,
            ROW_LABELS[model],
            va="center",
            ha="center",
            rotation="vertical",
            transform=axs[i, 0].transAxes,
            fontweight="bold",
        )

    axs[0, 0].set_title(r"W $-$ C")
    axs[0, 1].set_title(r"W $+$ C")

    fig.colorbar(
        im, ax=list(axs.ravel()), orientation="horizontal", fraction=0.04, pad=0.05, extend="both", label="SSTA [K]"
    )

    gpl.enumerate_axes(axs, pos_x=0.01, pos_y=0.95, fontsize="medium")
    save_figure(fig, f"fig7_enso_asymmetry_lag{tau}_month{init_month}.pdf", args.output)


if __name__ == "__main__":
    main()
