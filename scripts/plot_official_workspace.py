"""Plot aggregate results from the released Anthropic workspace prompt sets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evals.judge_workspace_readouts import summarise_overall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    source = Path(args.judged)
    data = json.loads(source.read_text())
    overall = data.get("summary_overall") or summarise_overall(
        data["summary"], data["summary_band"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    copied = out / "data" / "normal_future_lens_workspace_full_judged.json"
    if source.resolve() != copied.resolve():
        shutil.copy2(source, copied)
    (out / "data" / "official_workspace_overall.json").write_text(
        json.dumps(overall, indent=2)
    )

    colors = {"raw": "#4c78a8", "jac": "#59a14f", "shuffle": "#bab0ac"}
    layers = ("20", "32", "42", "54", "62")
    x = np.arange(len(layers))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode in ("raw", "jac", "shuffle"):
        values = [
            overall["by_layer"][mode][layer]["concept_recall_at_k"]
            for layer in layers
        ]
        axes[0].plot(x, values, marker="o", linewidth=2, color=colors[mode],
                     label={"raw": "Raw residual", "jac": "Jacobian transported",
                            "shuffle": "Shuffled activation"}[mode])
    axes[0].set_xticks(x, [f"L{layer}" for layer in layers])
    axes[0].set_ylim(0, 0.38)
    axes[0].set_ylabel("Expected-concept recall")
    axes[0].set_title("Per-layer natural-language readout")
    axes[0].legend(frameon=False, fontsize=8)

    labels = ("Raw lens", "Jacobian lens", "J-lens top-20", "Shuffle")
    values = (
        overall["band"]["raw"]["concept_recall_across_band"],
        overall["band"]["jac"]["concept_recall_across_band"],
        overall["band"]["raw"]["jlens_concept_recall_across_band"],
        overall["band"]["shuffle"]["concept_recall_across_band"],
    )
    bars = axes[1].bar(np.arange(4), values,
                       color=(colors["raw"], colors["jac"], "#f28e2b", colors["shuffle"]))
    axes[1].set_xticks(np.arange(4), labels, rotation=18, ha="right")
    axes[1].set_ylim(0, 0.75)
    axes[1].set_ylabel("Recall at any of five layers")
    axes[1].set_title("Workspace-band recovery")
    axes[1].bar_label(bars, labels=[f"{100 * v:.1f}%" for v in values], padding=3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Ordinary future lens recovers activation-specific workspace content")
    fig.tight_layout()
    fig.savefig(out / "official_workspace.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "official_workspace.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
