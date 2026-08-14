"""Plot train and phrase-disjoint validation CE for the scaled repeat lens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.metrics).read_text().splitlines()]
    # A restarted run may append to an existing metrics file. Keep the final
    # monotonic step sequence so an abandoned startup does not duplicate the
    # beginning of the learning curve.
    resets = [i for i in range(1, len(rows))
              if int(rows[i]["step"]) <= int(rows[i - 1]["step"])]
    discarded_prefix_rows = resets[-1] if resets else 0
    rows = rows[discarded_prefix_rows:]
    steps = np.array([int(row["step"]) + 1 for row in rows])
    losses = np.array([float(row["loss"]) for row in rows])
    window = min(50, len(losses))
    kernel = np.ones(window) / window
    smooth = np.convolve(losses, kernel, mode="valid")
    smooth_steps = steps[window - 1:]
    heldout = [row for row in rows if "heldout_loss" in row]
    val_steps = np.array([int(row["step"]) + 1 for row in heldout])
    val_losses = np.array([float(row["heldout_loss"]) for row in heldout])
    best_idx = int(val_losses.argmin())

    out = Path(args.out_dir)
    (out / "data").mkdir(parents=True, exist_ok=True)
    curve = {
        "train": [{"step": int(s), "loss": float(v)} for s, v in zip(steps, losses)],
        "train_smoothed": [
            {"step": int(s), "loss": float(v)} for s, v in zip(smooth_steps, smooth)
        ],
        "validation": [
            {"step": int(s), "loss": float(v), "ppl": float(np.exp(v))}
            for s, v in zip(val_steps, val_losses)
        ],
        "best": {"step": int(val_steps[best_idx]), "loss": float(val_losses[best_idx])},
        "discarded_restart_prefix_rows": discarded_prefix_rows,
    }
    (out / "data" / "validation_curve.json").write_text(json.dumps(curve, indent=2))

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(smooth_steps, smooth, color="#999999", label=f"Train CE ({window}-step mean)")
    ax.plot(val_steps, val_losses, "o-", color="#d97757", label="Phrase-disjoint validation CE")
    ax.scatter([val_steps[best_idx]], [val_losses[best_idx]], color="#2f6f5e", zorder=3)
    ax.set_xlabel("Optimizer updates (batch size 16)")
    ax.set_ylabel("Token cross-entropy")
    ax.set_title(
        f"Phrase-disjoint validation CE reaches {val_losses[best_idx]:.3f} "
        f"at step {val_steps[best_idx]}"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "validation_curve.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "validation_curve.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
