"""Plot the futurelens data-scaling CROSS-ENTROPY curve (held-out, disjoint docs).

For each pretraining checkpoint and fed layer L:
  CE_L = -mean log p_lens(model's own continuation tokens)   [teacher-forced through the
         lens with layer L's residual injected]  -- lower = the lens more faithfully predicts
         what the model actually says next.
Floor H_model = the base model's self cross-entropy on its OWN temp-1 continuations
         (= the model's continuation entropy). gap = CE_L - H_model = KL(model||lens) nats.

Reads   ~/shared/reports/skiplens-scaling/data/scaling_ce.json
Writes  ~/shared/reports/skiplens-scaling/scaling_ce.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-scaling")
d = json.load(open(f"{RD}/data/scaling_ce.json"))
BPS = d.get("batch_per_step", 64)
FED = [62, 48, 34, 26, 18, 10]
H = d["heldout"]
cmap = plt.cm.viridis
colors = {l: cmap(i / (len(FED) - 1)) for i, l in enumerate(FED)}

its = sorted(int(x) for x in H)
xs = [it * BPS / 1000.0 for it in its]
floors = [H[str(it)].get("h_model") for it in its if H[str(it)].get("h_model") is not None]
floor = sum(floors) / len(floors) if floors else None

fig, ax = plt.subplots(figsize=(8.4, 5.6))
for l in FED:
    ys = [H[str(it)].get("ce", {}).get(str(l)) for it in its]
    xy = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if xy:
        ax.plot([p[0] for p in xy], [p[1] for p in xy], "-o", color=colors[l],
                ms=4.5, lw=1.9, label=f"fed L{l}")
if floor is not None:
    ax.axhline(floor, color="black", ls="--", lw=2.4,
               label=f"H_model floor = {floor:.2f} nats  (model's own continuation entropy)")

# annotate the gap at the deepest and penultimate layers, last checkpoint
if floor is not None and its:
    last = str(its[-1])
    for l, dy in [(62, 8), (10, -12)]:
        ce = H[last].get("ce", {}).get(str(l))
        if ce is not None:
            ax.annotate(f"L{l} gap {ce-floor:+.2f}", (xs[-1], ce),
                        textcoords="offset points", xytext=(6, dy), fontsize=8, color=colors[l])

ax.set_xlabel("Distinct pretraining examples seen (thousands)")
ax.set_ylabel("Cross-entropy of the model's continuation under the lens (nats)\nlower = lens more faithfully predicts what the model says next")
ax.set_title("Held-out next-token faithfulness: deeper fed layers cost strictly more excess\n"
             "cross-entropy over the model, and the gap is FLAT in data — faithfulness saturates early",
             fontsize=11.5, pad=16)
ax.legend(fontsize=8, ncol=2, loc="upper right", framealpha=0.92)
ax.grid(alpha=0.25)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/scaling_ce.{ext}", dpi=150, bbox_inches="tight")
print("wrote", f"{RD}/scaling_ce.png / .pdf", "| checkpoints", len(its),
      "| floor", (round(floor, 3) if floor else None))
