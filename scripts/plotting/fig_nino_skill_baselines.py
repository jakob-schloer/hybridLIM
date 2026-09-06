"""Figure 5: Skill of the deep-learning baselines.

Niño4 skill of CS-LIM, LIM-LSTM, LSTM and ConvLSTM, scored on the test set:
 (a, b) RMSESS / CRPSS over forecast lead time for models trained on the full
        1500-year training set;
 (c, d) RMSESS / CRPSS at a fixed lead time vs. number of training years.
All skill scores are relative to monthly climatology.

Where the registry lists several runs per training-data subset (repeated
trainings with different weight initialization and data shuffling), panels
(c, d) show the mean across runs with error bars giving their std.
"""

from utils import NUM_DATA
from utils import base_parser
from utils import display_label
from utils import load_experiments
from utils import load_nino_scores
from utils import load_nino_scores_ndata_runs
from utils import model_color
from utils import plt
from utils import save_figure

import hyblim.geoplot as gpl

SCORE_LABELS = {"rmsess": "RMSESS", "crpss": "CRPSS"}


def main():
    parser = base_parser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["LIM", "LIM+LSTM", "LSTM", "ConvLSTM"], help="Models to plot (registry keys)."
    )
    parser.add_argument("--scores", nargs="+", default=["rmsess", "crpss"])
    parser.add_argument("--lag", type=int, default=12, help="Lead time for the training-data panels (c, d).")
    parser.add_argument("--index", type=str, default="nino4")
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)

    print("Generating Figure 5 (deep-learning baselines)...")
    # Top row: full-data models scored over lead time.
    scores_full, _ = load_nino_scores(experiments, args.models, args.datasplit)
    # Bottom row: training-data sweep, scored at a fixed lead time; all repeated
    # runs are kept so the panels can show their mean and spread.
    scores_ndata, _ = load_nino_scores_ndata_runs(experiments, args.models, NUM_DATA, args.datasplit)

    ncols = len(args.scores)
    fig, axs = plt.subplots(2, ncols, figsize=(4 * ncols, 6))

    # --- Panels (a, b): skill over lead time ---
    for i, score_name in enumerate(args.scores):
        ax = axs[0, i]
        for model in args.models:
            if model not in scores_full or score_name not in scores_full[model]:
                continue
            score = scores_full[model][score_name][args.index]
            ax.plot(score["lag"], score, "-", color=model_color(experiments, model), label=display_label(model))
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$\tau$ [months]")
        ax.set_ylabel(rf"{SCORE_LABELS.get(score_name, score_name)} ({args.index})")
        ax.set_ylim(-0.1, 0.95)
        ax.set_xticks(score["lag"][::2])
        if i == 0:
            ax.legend(fontsize="small")

    # --- Panels (c, d): skill vs. number of training years ---
    for i, score_name in enumerate(args.scores):
        ax = axs[1, i]
        for model in args.models:
            if model not in scores_ndata or score_name not in scores_ndata[model]:
                continue
            score = scores_ndata[model][score_name][args.index].sel(lag=args.lag)
            ax.errorbar(
                score["ndata"],
                score.mean(dim="run", skipna=True).values,
                yerr=score.std(dim="run", skipna=True).values,
                marker="o",
                linestyle="-",
                capsize=3,
                color=model_color(experiments, model),
                label=display_label(model),
            )
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(NUM_DATA)
        ax.set_xticklabels([n // 12 for n in NUM_DATA], rotation=45)
        ax.set_xlabel("Number of training years")
        ax.set_ylabel(rf"{SCORE_LABELS.get(score_name, score_name)} " rf"({args.index}, $\tau={args.lag}$)")

    gpl.enumerate_axes(axs, pos_x=0.02, pos_y=0.97, fontsize="medium")
    fig.tight_layout()
    save_figure(fig, "fig5_nino_skill_baselines.pdf", args.output)


if __name__ == "__main__":
    main()
