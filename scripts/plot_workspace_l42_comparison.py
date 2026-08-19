"""Plot the paired repeat-trained versus ordinary future-lens comparison."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = (
    ("workspace_recall", "Expected workspace concept recall", "fraction"),
    ("joint_recovery", "All concepts in one readout", "fraction"),
    ("coherence", "Mean coherence", "score"),
    ("hallucination", "Unrelated-content rate", "fraction"),
)
COLORS = ("#4c78a8", "#d97757")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    source = Path(args.comparison)
    data = json.loads(source.read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    shutil.copy2(source, out / "data" / "repeat_workspace_comparison.json")

    layers = ("42", "62")
    x = np.arange(len(layers))
    width = 0.34
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.3))
    for ax, (metric, label, scale) in zip(axes.flat, METRICS):
        normal = [data["summary"]["all"][layer][metric]["normal"] for layer in layers]
        repeat = [data["summary"]["all"][layer][metric]["repeat"] for layer in layers]
        left = ax.bar(x - width / 2, normal, width, label="Ordinary future lens",
                      color=COLORS[0])
        right = ax.bar(x + width / 2, repeat, width, label="Repeat-only lens",
                       color=COLORS[1])
        ax.set_xticks(x, [f"Feed layer {layer}" for layer in layers])
        ax.set_ylabel(label)
        ax.spines[["top", "right"]].set_visible(False)
        if scale == "fraction":
            ax.set_ylim(0, min(1.0, max(normal + repeat) * 1.35 + 0.03))
            ax.bar_label(left, fmt="%.1f%%", labels=[f"{100 * v:.1f}%" for v in normal],
                         padding=3, fontsize=8)
            ax.bar_label(right, fmt="%.1f%%", labels=[f"{100 * v:.1f}%" for v in repeat],
                         padding=3, fontsize=8)
        else:
            ax.set_ylim(0, 5)
            ax.bar_label(left, fmt="%.2f", padding=3, fontsize=8)
            ax.bar_label(right, fmt="%.2f", padding=3, fontsize=8)
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Repeat-only training does not produce coherent layer-42 workspace readouts",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / "repeat_workspace_comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "repeat_workspace_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
