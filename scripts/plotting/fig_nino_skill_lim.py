"""Figure 2: RMSE and CRPS skill scores of the LIM versions and LIM-LSTM.

Niño4 skill over forecast lead time on the test set, showing the LIM-version
progression (ST-LIM ssta -> CS-LIM ssta -> CS-LIM ssta,ssha) and the LIM-LSTM
hybrid, with persistence as a reference. Panel (a) RMSESS, panel (b) CRPSS;
both relative to monthly climatology.
"""

import numpy as np
from utils import base_parser
from utils import gpl
from utils import load_experiments
from utils import load_nino_scores
from utils import model_color
from utils import plt
from utils import save_figure

# Registry key -> manuscript legend label (keys without an entry use the key).
LABELS = {
    "LIM": "CS-LIM (ssta,ssha)",
    "LIM+LSTM": "LIM-LSTM",
}
SCORE_LABELS = {"rmsess": "RMSESS", "crpss": "CRPSS"}


def main():
    parser = base_parser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ST-LIM (ssta)", "CS-LIM (ssta)", "LIM", "LIM+LSTM"],
        help="LIM variants / hybrid to plot (registry keys).",
    )
    parser.add_argument(
        "--reference", type=str, default="Persistence", help="Reference model drawn as a dashed line (or 'none')."
    )
    parser.add_argument("--scores", nargs="+", default=["rmsess", "crpss"])
    parser.add_argument("--index", type=str, default="nino4")
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)

    print("Generating Figure 2 (LIM-version & hybrid skill)...")
    models = list(args.models)
    if args.reference.lower() != "none":
        models.append(args.reference)
    scores, _ = load_nino_scores(experiments, models, args.datasplit)

    ncols = len(args.scores)
    fig, axs = plt.subplots(1, ncols, figsize=(8, 3.0), sharex=True)
    axs = np.atleast_1d(axs)

    for i, score_name in enumerate(args.scores):
        ax = axs[i]
        for model in args.models:
            if model not in scores or score_name not in scores[model]:
                continue
            score = scores[model][score_name][args.index]
            ax.plot(score["lag"], score, "-", color=model_color(experiments, model), label=LABELS.get(model, model))

        # Reference (persistence): dashed, only where the score exists.
        ref = args.reference
        if ref.lower() != "none" and ref in scores and score_name in scores[ref]:
            score = scores[ref][score_name][args.index]
            ax.plot(score["lag"], score, "--", color=model_color(experiments, ref), label=LABELS.get(ref, ref))

        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_xlabel(r"$\tau$ [months]")
        ax.set_ylabel(rf"{SCORE_LABELS.get(score_name, score_name)} ({args.index})")
        if i == 0:
            ax.legend(fontsize="small")

    axs[-1].set_ylim(-0.1, 0.95)
    axs[-1].set_xticks(score["lag"][::2])
    gpl.enumerate_axes(axs, pos_x=0.01, pos_y=1.05, fontsize="medium")

    save_figure(fig, "fig2_nino_skill_lim.pdf", args.output)


if __name__ == "__main__":
    main()
