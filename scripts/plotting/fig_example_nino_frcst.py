"""Figure 4: Example El Niño forecast.

A forecast initialized 12 months before an El Niño in the test set.
 (a) Niño4 trajectory over lead time: target, CS-LIM and LIM-LSTM ensemble
     mean (line) and spread (shading), with a dashed marker at the 12-month lead.
 (b-d) Mean SSTA forecast (color) and SSHA (contours) at tau=12 months for the
     CS-LIM (b), LIM-LSTM (c) and target (d).
"""

import cartopy as ctp
import cftime
import matplotlib.gridspec as gridspec
import numpy as np
import xarray as xr
from utils import base_parser
from utils import display_label
from utils import enso
from utils import gpl
from utils import load_experiments
from utils import load_raw_data_and_eof
from utils import model_color
from utils import plt
from utils import preproc
from utils import save_figure

HINDCAST_FILE = {
    "LIM": "cslim_hindcast_ssta-ssha_eof20-10_{split}.nc",
    "LIM+LSTM": "limlstm_hindcast_ssta-ssha_eof20-10_{split}.nc",
}
SSTA_ARGS = dict(cmap="RdBu_r", vmin=-3, vmax=3, eps=0.5, centercolor="white")
SSHA_ARGS = dict(
    kwargs_pl={"colors": "k", "levels": [-1.5, -0.75, 0.75, 1.5], "linewidths": 1.0},
    zerolinecolor=None,
    add_inline_labels=False,
)


def reconstruct_members(hindcast, combined_eof, normalizer, init_time, lag):
    """Per-member SSTA/SSHA grid forecast (in physical units) at (init, lag)."""
    z = hindcast["z"].sel(time=init_time, lag=lag).values  # (member, eof)
    rec = combined_eof.reconstruction(z, times=np.arange(z.shape[0]))  # 'time' = member
    rec["ssta"] = normalizer.inverse_transform(rec["ssta"])
    return rec


def nino_trajectory(hindcast, combined_eof, normalizer, init_time, lags, index):
    """Ensemble mean and std of a Niño index over lead time (physical units)."""
    mean, std = [], []
    for lag in lags:
        rec = reconstruct_members(hindcast, combined_eof, normalizer, init_time, lag)
        n = enso.get_nino_indices(rec["ssta"])[index]  # one value per member
        mean.append(float(n.mean()))
        std.append(float(n.std()))
    return np.array(mean), np.array(std)


def _ensemble_nino_at_lag(hindcast, init_times, lag, combined_eof, normalizer, index):
    """Ensemble-mean Niño index at a single lead for many initializations (batched)."""
    z = np.stack([hindcast["z"].sel(time=i, lag=lag).values for i in init_times])  # (N, M, E)
    n, m, e = z.shape
    rec = combined_eof.reconstruction(z.reshape(n * m, e), times=np.arange(n * m))
    ssta = normalizer.inverse_transform(rec["ssta"])
    return enso.get_nino_indices(ssta)[index].values.reshape(n, m).mean(axis=1)


def select_event(
    target_lim, frcst_lim, hindcasts, combined_eof, normalizer, index, lag, peak_thresh=1.3, init_thresh=0.3
):
    """Select strong El Niño."""
    obs = target_lim[index].sel(lag=1)  # observed series (valid time)
    frc = frcst_lim[index].mean("member")  # CS-LIM forecast (valid time)

    def diag(init, da, L):
        v = preproc.add_to_cftime([init], n_month=L)[0]
        s = da.sel(lag=L)
        return float(s.sel(time=v).values) if (s["time"].values == v).any() else np.nan

    cands, peaks = [], []
    for init in hindcasts["LIM+LSTM"]["time"].values:
        t_peak = preproc.add_to_cftime([init], n_month=lag)[0]
        if not (obs["time"].values == t_peak).any() or not (obs["time"].values == init).any():
            continue
        peak = float(obs.sel(time=t_peak).values)
        if peak < peak_thresh or float(obs.sel(time=init).values) > init_thresh:
            continue
        shape = [diag(init, frc, L) for L in [lag - 6, lag - 3, lag, lag + 3]]
        if np.nanargmax(shape) != 2:
            continue
        cands.append(init)
        peaks.append(peak)
    if not cands:
        raise RuntimeError("No developing El Niño matched the selection criteria.")

    peaks = np.array(peaks)
    return cands[int(np.argmax(peaks))]


def main():
    parser = base_parser(description=__doc__)
    parser.add_argument("--lag", type=int, default=12, help="Lead time for the maps [months].")
    parser.add_argument("--index", type=str, default="nino4")
    parser.add_argument(
        "--init", type=str, default=None, help="Initialization time 'YYYY-MM' (default: auto-selected event)."
    )
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)
    lim_dir = experiments["LIM"]["paths"][0]
    hyb_dir = experiments["LIM+LSTM"]["paths"][0]
    idx_name, lag = args.index, args.lag

    print("Generating Figure 4 (example El Niño forecast)...")
    raw = load_raw_data_and_eof()
    combined_eof, normalizer = raw["combined_eof"], raw["normalizers"]["ssta"]

    frcst_lim = xr.open_dataset(lim_dir / "metrics" / f"nino_frcst_{args.datasplit}.nc")
    target_lim = xr.open_dataset(lim_dir / "metrics" / f"nino_target_{args.datasplit}.nc")
    lags = np.sort(target_lim[idx_name]["lag"].values.astype(int))
    hindcasts = {
        "LIM": xr.open_dataset(lim_dir / HINDCAST_FILE["LIM"].format(split=args.datasplit)),
        "LIM+LSTM": xr.open_dataset(hyb_dir / HINDCAST_FILE["LIM+LSTM"].format(split=args.datasplit)),
    }

    # 1817-03-01, 1968-02-01, 1869-03-01
    if args.init is None:
        init_time = select_event(target_lim, frcst_lim, hindcasts, combined_eof, normalizer, idx_name, lag)
    else:
        year, month = map(int, args.init.split("-"))
        init_time = cftime.DatetimeNoLeap(year, month, 1)
    print(f"  initialization: {init_time} (El Niño peak at lag {lag})")

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.1, 1.0], hspace=0.35, wspace=0.28)

    # --- Panel (a): Niño4 trajectory over lead time ---
    content_axes = []
    ax = fig.add_subplot(gs[:, 0])
    content_axes.append(ax)

    # Target trajectory = observed Niño4 at the valid times init+L (diagonal).
    obs = target_lim[idx_name].sel(lag=1)
    tgt = []
    for L in lags:
        v = preproc.add_to_cftime([init_time], n_month=int(L))[0]
        tgt.append(float(obs.sel(time=v).values) if (obs["time"].values == v).any() else np.nan)
    ax.plot(lags, tgt, "k-", lw=2, label="Target", zorder=5)

    for model in ["LIM", "LIM+LSTM"]:
        mean, std = nino_trajectory(hindcasts[model], combined_eof, normalizer, init_time, lags, idx_name)
        clr = model_color(experiments, model)
        ax.plot(lags, mean, "-", color=clr, label=display_label(model))
        ax.fill_between(lags, mean - std, mean + std, color=clr, alpha=0.2)

    ax.axvline(lag, color="grey", ls="--", lw=1)
    ax.axhline(0, color="grey", ls="-", lw=0.5)
    ax.set_xlabel(r"Lead time $\tau$ [months]")
    ax.set_ylabel(f"{idx_name.capitalize()} index [K]")
    ax.set_xticks(lags)
    ax.legend(fontsize="small")

    # --- Panels (b-d): SSTA (color) + SSHA (contour) maps at tau=lag ---
    proj = ctp.crs.PlateCarree(central_longitude=180)
    valid_time = preproc.add_to_cftime([init_time], n_month=lag)[0]

    fields = {
        "LIM": reconstruct_members(hindcasts["LIM"], combined_eof, normalizer, init_time, lag).mean("time"),
        "LIM+LSTM": reconstruct_members(hindcasts["LIM+LSTM"], combined_eof, normalizer, init_time, lag).mean("time"),
        "Target": raw["ds"]
        .sel(time=valid_time)
        .assign(ssta=normalizer.inverse_transform(raw["ds"]["ssta"].sel(time=valid_time))),
    }

    im = target_ax = None
    for row, name in enumerate(["LIM", "LIM+LSTM", "Target"]):
        field = fields[name]
        ax = fig.add_subplot(gs[row, 1], projection=proj)
        content_axes.append(ax)
        out = gpl.plot_map(field["ssta"], ax=ax, central_longitude=180, add_bar=False, **SSTA_ARGS)
        im = out["im"]
        gpl.plot_contour(field["ssha"], ax=ax, central_longitude=180, **SSHA_ARGS)
        ax.set_title("Target" if name == "Target" else rf"{display_label(name)} ($\tau={lag}$)", fontsize="small")
        out["gl"].bottom_labels = row == 2
        target_ax = ax

    # Horizontal colorbar in its own axes below the target map, matching its width
    # (a dedicated axes keeps all three maps the same size).
    fig.set_layout_engine("none")  # disable autolayout (incompatible with GeoAxes)
    fig.canvas.draw()
    pos = target_ax.get_position()
    cax = fig.add_axes([pos.x0, pos.y0 - 0.07, pos.width, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label("SSTA [K]")

    gpl.enumerate_axes(content_axes, pos_x=0.01, pos_y=0.97, fontsize="medium")
    save_figure(fig, f"fig4_example_nino_frcst_lag{lag}.pdf", args.output)


if __name__ == "__main__":
    main()
