"""Plot the full scaled FineWeb OPD/SFT validation trajectory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"opd": "#4c78a8", "sft_matched": "#f28e2b"}
LABELS = {"opd": "Reverse-KL OPD", "sft_matched": "Matched SFT"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    source = Path(args.analysis)
    data = json.loads(source.read_text())
    out = Path(args.out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    target = out / "data" / "scaled_opd_analysis.json"
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)

    steps = data["common_steps"]
    metric_specs = (
        ("reference_student_nll", "Reference NLL ↓"),
        ("reference_mean_reverse_kl", "Exact reverse KL ↓"),
        ("reference_top1_agreement", "Teacher top-1 agreement ↑"),
        ("mean_length", "Greedy continuation length /8"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for ax, (metric, label) in zip(axes.flat, metric_specs):
        for arm in ("opd", "sft_matched"):
            for feed, linestyle in (("L62", "-"), ("L42", "--")):
                values = [
                    data["trajectory"][feed][arm][str(step)]["aggregate"][metric]
                    for step in steps
                ]
                ax.plot(
                    steps, values, color=COLORS[arm], linestyle=linestyle,
                    marker="o", ms=3,
                    label=f"{LABELS[arm]} {feed}",
                )
        ax.set_xlabel("Stage-two optimizer update")
        ax.set_ylabel(label)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    selected = data["selection"]["selected_opd_step"]
    delta = data["paired_comparisons"]["L62"]["reference_student_nll"][
        "opd_minus_sft"
    ]
    fig.suptitle(
        f"Best OPD L62 checkpoint is at update {selected}; "
        f"its NLL gap to matched SFT is {delta:+.3f} nats/token"
    )
    fig.tight_layout()
    fig.savefig(out / "scaled_opd_validation.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "scaled_opd_validation.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
