"""Figure 1: Forecast skill vs. training-data length.

Anomaly correlation coefficient (ACC, = `cc`) of the Nino4 index at a fixed
forecast lead time, evaluated on the test set, as a function of the number of
training years. One curve per model (CS-LIM, LIM-LSTM, LSTM).
"""

from utils import NUM_DATA
from utils import base_parser
from utils import load_experiments
from utils import load_nino_scores_ndata
from utils import model_color
from utils import plt
from utils import save_figure


def main():
    parser = base_parser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", default=["LIM", "LIM+LSTM", "LSTM"], help="Models to plot (base names in the registry)."
    )
    parser.add_argument("--lag", type=int, default=12, help="Forecast lead time in months (default: 12).")
    parser.add_argument("--index", type=str, default="nino4", help="Nino index to score (default: nino4).")
    args = parser.parse_args()

    experiments = load_experiments(args.experiments)

    print("Generating Figure 1 (skill vs. training-data length)...")
    scores, _ = load_nino_scores_ndata(experiments, args.models, NUM_DATA, args.datasplit)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    for model in args.models:
        if model not in scores or "cc" not in scores[model]:
            continue
        acc = scores[model]["cc"][args.index].sel(lag=args.lag)
        ax.plot(acc["ndata"], acc.values, marker="o", linestyle="-", color=model_color(experiments, model), label=model)

    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8, label="skillful (ACC=0.5)")

    ax.set_xscale("log")
    ax.set_xticks(NUM_DATA)
    ax.set_xticklabels([n // 12 for n in NUM_DATA], rotation=45)
    ax.set_xlabel("Number of training years")
    ax.set_ylabel(rf"ACC ({args.index}, $\tau={args.lag}$ months)")
    ax.legend(loc="lower right", fontsize="small")

    save_figure(fig, "fig1_skill_ntrain.pdf", args.output)


if __name__ == "__main__":
    main()
