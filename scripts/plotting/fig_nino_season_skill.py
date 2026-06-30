"""Appendix Figure A1: Seasonal dependency of the Niño-index forecast skill.

- left column  : the CS-LIM seasonal skill (RMSESS vs. climatology).
- right column : the skill *difference* of the LIM-LSTM relative to the CS-LIM
                 (red -> hybrid improves on the linear model).
"""

import numpy as np
from utils import base_parser
from utils import display_label
from utils import gpl
from utils import load_experiments
from utils import load_nino_scores
from utils import plt
from utils import save_figure

MONTH_LETTERS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

# Absolute-skill (left) and skill-difference (right) color specs per metric.
ABS_ARGS = {"rmsess": dict(cmap="plasma", vmin=0, vmax=0.9), "crpss": dict(cmap="plasma_r", vmin=0, vmax=0.6)}
DIFF_ARGS = {
    "rmsess": dict(cmap="RdBu_r", vmin=-0.2, vmax=0.2, eps=0.05, centercolor="#FFFFFF"),
    "crpsss": dict(cmap="RdBu_r", vmin=-0.1, vmax=0.1, eps=0.01, centercolor="#FFFFFF"),
}


def main():
    parser = base_parser(description=__doc__)
    parser.add_argument("--ctrl", type=str, default="LIM", help="Baseline model (left column + difference reference).")
    parser.add_argument(
        "--model", type=str, default="LIM+LSTM", help="Model compared against the baseline (right column)."
    )
    parser.add_argument("--metric", type=str, default="rmsess", help="Score to plot (absolute, left column).")
    parser.add_argument(
        "--indices", nargs="+", default=["nino4"], help="Niño regions to show (one row each, west -> east)."
    )
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)
    metrickey = args.metric

    print("Generating Appendix Figure A1 (seasonal Niño skill)...")
    _, scores_month = load_nino_scores(experiments, [args.ctrl, args.model], args.datasplit)
    if args.ctrl not in scores_month or args.model not in scores_month:
        raise RuntimeError(f"Missing nino scores for {args.ctrl} / {args.model}.")

    nrows = len(args.indices)
    fig, axs = plt.subplots(nrows, 2, figsize=(9, nrows * 4), sharex=True, sharey=True)
    axs = np.atleast_2d(axs)

    for i, nino_idx in enumerate(args.indices):
        ctrl = scores_month[args.ctrl][metrickey][nino_idx]
        diff = scores_month[args.model][metrickey][nino_idx] - ctrl

        # Left: absolute CS-LIM seasonal skill.
        gpl.plot_matrix(
            ctrl,
            "lag",
            "month",
            ax=axs[i, 0],
            add_bar=True,
            **ABS_ARGS[metrickey],
            kwargs_cb={"orientation": "horizontal", "extend": "both", "label": metrickey.upper()},
        )
        # Right: hybrid - baseline difference.
        gpl.plot_matrix(
            diff,
            "lag",
            "month",
            ax=axs[i, 1],
            add_bar=True,
            **DIFF_ARGS[metrickey],
            kwargs_cb={"orientation": "horizontal", "extend": "both", "label": rf"$\Delta$ {metrickey.upper()}"},
        )

        for ax in (axs[i, 0], axs[i, 1]):
            ax.set_yticks(ctrl["month"].values)
            ax.set_xticks(ctrl["lag"].values)
        axs[i, 0].set_yticklabels(MONTH_LETTERS)
        axs[i, 0].set_ylabel(nino_idx.capitalize())

    for ax in axs[-1, :]:
        ax.set_xlabel(r"$\tau$ [month]")
    axs[0, 0].set_title(display_label(args.ctrl))
    axs[0, 1].set_title(f"{display_label(args.model)} $-$ {display_label(args.ctrl)}")

    # fig.colorbar(im_abs, ax=list(axs[:, 0]), location="bottom",
    #              fraction=0.1, pad=0.3, extend="max", label=metrickey.upper())
    # fig.colorbar(im_diff, ax=list(axs[:, 1]), location="bottom",
    #              fraction=0.1, pad=0.3, extend="both",
    #              label=rf"$\Delta$ {metrickey.upper()}")

    gpl.enumerate_axes(axs, pos_x=0.02, pos_y=0.95, fontsize="medium")
    save_figure(fig, f"figA1_compare_nino_season_{metrickey}.pdf", args.output)


if __name__ == "__main__":
    main()
