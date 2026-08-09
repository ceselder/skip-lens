"""Plot corrected reverse-KL OPD against token-matched SFT."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    source = Path(args.analysis)
    data = json.loads(source.read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    copied = out / "data" / "true_opd8_analysis.json"
    if source.resolve() != copied.resolve():
        shutil.copy2(source, copied)

    conditions = (
        ("normal", "L62", "Normal\nL62"),
        ("normal", "L42", "Normal\nL42"),
        ("repeat", "L62", "Repeat\nL62"),
        ("repeat", "L42", "Repeat\nL42"),
    )
    metrics = (
        ("nll", "Reference NLL ↓"),
        ("reverse_kl", "Exact reverse KL ↓"),
        ("coherence", "Coherence /5 ↑"),
        ("hallucination", "Hallucination rate ↓"),
    )

    def value(dataset: str, feed: str, arm: str, metric: str) -> float:
        if metric == "nll":
            return data["evaluations"][dataset][feed][arm]["reference_student_nll"]
        if metric == "reverse_kl":
            return data["evaluations"][dataset][feed][arm][
                "reference_mean_reverse_kl"]
        quality = data["quality"]["by_condition"][dataset][arm][feed]
        if metric == "coherence":
            return quality["coherence"]
        return quality["hallucination_rate"]

    x = np.arange(len(conditions))
    width = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for ax, (metric, title) in zip(axes.flat, metrics):
        opd = [value(dataset, feed, "opd", metric) for dataset, feed, _ in conditions]
        sft = [value(dataset, feed, "sft", metric) for dataset, feed, _ in conditions]
        left = ax.bar(x - width / 2, opd, width, label="Reverse-KL OPD",
                      color="#4c78a8")
        right = ax.bar(x + width / 2, sft, width, label="Token-matched SFT",
                       color="#f28e2b")
        ax.set_xticks(x, [label for _, _, label in conditions])
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "coherence":
            ax.set_ylim(0, 5)
        elif metric == "hallucination":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, max(opd + sft) * 1.25)
        labels_opd = [f"{v:.2f}" for v in opd]
        labels_sft = [f"{v:.2f}" for v in sft]
        if metric == "hallucination":
            labels_opd = [f"{100 * v:.0f}%" for v in opd]
            labels_sft = [f"{100 * v:.0f}%" for v in sft]
        ax.bar_label(left, labels=labels_opd, padding=3, fontsize=8)
        ax.bar_label(right, labels=labels_sft, padding=3, fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("True on-policy reverse-KL distillation versus matched SFT")
    fig.tight_layout()
    fig.savefig(out / "true_opd8_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "true_opd8_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
