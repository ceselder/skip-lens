"""Plot the futurelens data-scaling skip-lens curve: FVE per fed layer vs the number
of distinct pretraining examples seen. One line per fed layer + a bold mean line;
leaked eval drawn solid, held-out (disjoint-doc) eval overlaid dashed when present.

Reads   ~/shared/reports/skiplens-scaling/data/scaling_fve.json
Writes  ~/shared/reports/skiplens-scaling/scaling_fve.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-scaling")
d = json.load(open(f"{RD}/data/scaling_fve.json"))
BPS = d.get("batch_per_step", 64)
FED = [62, 48, 34, 26, 18, 10]                      # penultimate -> deep
cmap = plt.cm.viridis
colors = {l: cmap(i / (len(FED) - 1)) for i, l in enumerate(FED)}


def series(split, layer):
    pts = sorted((int(it), rec) for it, rec in d[split].items())
    xs = [it * BPS / 1000.0 for it, _ in pts]                        # examples seen (thousands)
    ys = [100 * rec["per_layer"].get(str(layer), rec["per_layer"].get(layer))
          if (rec["per_layer"].get(str(layer)) is not None
              or rec["per_layer"].get(layer) is not None) else None
          for _, rec in pts]
    return xs, ys


def mean_series(split):
    pts = sorted((int(it), rec) for it, rec in d[split].items())
    return ([it * BPS / 1000.0 for it, _ in pts],
            [100 * rec["mean"] if rec["mean"] is not None else None for _, rec in pts])


fig, ax = plt.subplots(figsize=(8.2, 5.4))
ax.axhline(0, color="#999", lw=1, ls=":", zorder=1,
           label="predict-the-mean baseline (FVE=0)")

for split, ls, alpha, lab in [("leaked", "-", 1.0, "leaked"), ("heldout", "--", 0.9, "held-out")]:
    if not d[split]:
        continue
    for l in FED:
        xs, ys = series(split, l)
        xy = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not xy:
            continue
        ax.plot([p[0] for p in xy], [p[1] for p in xy], ls, color=colors[l],
                marker="o", ms=4, lw=1.6, alpha=alpha,
                label=(f"fed L{l}" if split == "leaked" else None))
    mx, my = mean_series(split)
    mxy = [(x, y) for x, y in zip(mx, my) if y is not None]
    if mxy:
        ax.plot([p[0] for p in mxy], [p[1] for p in mxy], ls, color="black",
                marker="s", ms=5, lw=2.6, alpha=alpha,
                label=(f"MEAN across fed layers ({lab})"))

ax.set_xlabel("Distinct pretraining examples seen (thousands)")
ax.set_ylabel("Skip-lens FVE (%)  —  fraction of L62 variance the lens reconstructs")
ax.set_title("More pretraining data lifts DEEP-layer skip-lens reconstruction\n"
             "(fed L10: −42%→−20%) while the penultimate layer (L62) saturates immediately",
             fontsize=11.5, pad=34)
title2 = "leaked eval only (pretrain saw these ctx docs)" if not d["heldout"] \
    else "solid = leaked (pretrain-seen docs);  dashed = held-out (disjoint docs)"
ax.text(0.5, 1.012, title2, transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9, color="#555")
ax.legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/scaling_fve.{ext}", dpi=150, bbox_inches="tight")
print("wrote", f"{RD}/scaling_fve.png / .pdf",
      "| leaked", len(d["leaked"]), "heldout", len(d["heldout"]))
