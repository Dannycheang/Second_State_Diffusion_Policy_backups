import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path: Path):
    epochs = []
    train_loss = []
    eval_epochs = []
    eval_success = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            epoch = int(float(row.get("epoch", 0)))
            epochs.append(epoch)
            train_loss.append(float(row.get("train_loss", 0.0)))
            if row.get("eval_success_rate") not in (None, ""):
                eval_epochs.append(epoch)
                eval_success.append(float(row["eval_success_rate"]))
    return epochs, train_loss, eval_epochs, eval_success


def main() -> None:
    root = Path(__file__).resolve().parent
    series = [
        ("PickCube final", root / "output_pickcube_state_final" / "metrics.csv"),
        ("Peg first round", root / "output_peg_state_motionplanning" / "metrics.csv"),
        ("Peg resume baseline", root / "output_peg_state_motionplanning_resume" / "metrics.csv"),
        ("Peg combined failed", root / "output_peg_state_combined" / "metrics.csv"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plotted = 0
    for label, metrics_path in series:
        if not metrics_path.exists():
            continue
        epochs, train_loss, eval_epochs, eval_success = read_metrics(metrics_path)
        if not epochs:
            continue
        axes[0].plot(epochs, train_loss, label=label, linewidth=1.6)
        if eval_epochs:
            axes[1].plot(eval_epochs, eval_success, label=label, linewidth=1.6)
        plotted += 1

    if plotted == 0:
        raise RuntimeError("No valid metrics found to plot.")

    axes[0].set_title("Train Loss Comparison")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Eval Success Rate Comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Success Rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    axes[0].legend()
    axes[1].legend()
    fig.tight_layout()

    out_path = root / "benchmark_results" / "training_curves_compare.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
