"""Plot the fixed-eight-token OPD comparison from saved evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ARMS = ("warm", "opd8", "sft8_tokens", "sft8_time")
LABELS = (
    "SFT warm start", "Forward KL (retired)", "SFT, equal tokens", "SFT, equal time"
)
COLORS = ("#9c948b", "#d97757", "#4c78a8", "#72a276")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    analysis = json.loads(Path(args.analysis).read_text())
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    normal = analysis["evaluations"]["normal"]["L62"]
    nll = [normal[arm]["reference_student_nll"] for arm in ARMS]
    kl = [normal[arm]["reference_mean_kl"] for arm in ARMS]
    top1 = [normal[arm]["reference_top1_agreement"] for arm in ARMS]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, values, ylabel in zip(
        axes, (nll, kl), ("Ground-truth NLL (nats/token)", "Teacher–student KL (nats)"),
    ):
        bars = ax.bar(range(len(ARMS)), values, color=COLORS)
        ax.set_xticks(range(len(ARMS)), LABELS, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylim(0, max(values) * 1.2)
    fig.suptitle(
        "The retired forward-KL ablation improves distribution fidelity, not validation loss",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "primary_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "primary_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    horizon = {}
    for arm in ("opd8", "sft8_tokens"):
        data = json.loads((eval_dir / f"normal_{arm}_L62_eval.json").read_text())
        horizon[arm] = {
            position: {
                "reference_student_nll": row["reference_student_nll"],
                "reference_mean_kl": row["reference_mean_kl"],
                "reference_top1_agreement": row["reference_top1_agreement"],
            }
            for position, row in data["by_horizon"].items()
        }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for arm, label, color in zip(
        ("opd8", "sft8_tokens"), ("Forward KL (retired)", "SFT, equal tokens"),
        COLORS[1:3]
    ):
        positions = [int(x) for x in horizon[arm]]
        axes[0].plot(
            positions,
            [horizon[arm][str(x)]["reference_student_nll"] for x in positions],
            marker="o", label=label, color=color,
        )
        axes[1].plot(
            positions,
            [horizon[arm][str(x)]["reference_mean_kl"] for x in positions],
            marker="o", label=label, color=color,
        )
    for ax, ylabel in zip(
        axes, ("Ground-truth NLL (nats/token)", "Teacher–student KL (nats)"),
    ):
        ax.set_xlabel("Continuation token position")
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(1, 9))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("The KL advantage persists across all eight continuation positions", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "horizon_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / "horizon_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    plot_data = {
        "aggregate": {
            arm: {"reference_student_nll": nll[i], "reference_mean_kl": kl[i],
                  "reference_top1_agreement": top1[i]}
            for i, arm in enumerate(ARMS)
        },
        "by_horizon": horizon,
    }
    (out_dir / "data" / "primary_plot.json").write_text(json.dumps(plot_data, indent=2))


if __name__ == "__main__":
    main()
