"""Per-layer residual-add dynamics (Qwen3.6-27B, 353 items, last token).
Left: mean norm of each layer's add ‖Δ_L‖ vs the running-stream norm ‖h_L‖ (log y).
Right: does each layer amplify or shrink the PREVIOUS stream — g_L=(Δ_L·ĥ_{L-1})/‖h_{L-1}‖ (signed)
and cos(Δ_L, h_{L-1}). Early layers amplify (write along the stream), the mid-stack adds orthogonally
(g≈0 — new content, no shrinkage), and the final layer L63 erases (g≪0). Reads data/layer_adds.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RD = os.path.expanduser("~/shared/reports/skiplens-layer-ablation")
g = json.load(open(f"{RD}/data/layer_adds.json"))
NLp1 = g["n_hidden_states"]
lab = g["labels"]
xs = list(range(NLp1))
hN = g["h_norm"]; dN = g["add_norm"]; sg = g["shrinkage_g"]; dc = g["add_stream_cos"]
CLAY, BLUE, GRN, MUT, GRID, INK = "#D97757", "#4C72B0", "#3aa06a", "#999999", "#e6e4dd", "#333333"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.8, 5.3))
fig.subplots_adjust(top=0.82, bottom=0.14, left=0.06, right=0.985, wspace=0.20)

axL.plot(xs, hN, "-o", color=BLUE, lw=2, ms=3, label="‖h$_L$‖  running-stream norm")
axL.plot(xs[1:], dN[1:], "-s", color=CLAY, lw=2, ms=3, label="‖Δ$_L$‖  norm of the layer's add")
axL.set_yscale("log"); axL.set_ylabel("norm (log)", fontsize=9.8, color=MUT)
axL.set_title("How much each layer adds", fontsize=11.5, color=INK, pad=6)

axR.axhline(0, color=INK, lw=0.8)
axR.plot(xs[1:], sg[1:], "-o", color=CLAY, lw=2, ms=3, label="g$_L$: amplify(+) / shrink(−) prior stream")
axR.plot(xs[1:], dc[1:], "-^", color=GRN, lw=1.7, ms=3, label="cos(Δ$_L$, h$_{L-1}$)  add-vs-stream")
axR.set_ylabel("signed (amplify + / shrink −)", fontsize=9.8, color=MUT)
axR.set_title("Does the layer amplify, add orthogonally, or shrink the stream?", fontsize=11, color=INK, pad=6)
axR.annotate("L63 erases", xy=(64, sg[64]), xytext=(52, -0.55), fontsize=9, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK, lw=1))
axR.axvspan(13, 55, color="#eef4ee", alpha=0.7, zorder=0)
axR.text(34, 0.42, "mid-stack: g≈0, cos≈0\n(orthogonal new content)", fontsize=8.2, color=INK, ha="center")

for ax in (axL, axR):
    ax.set_xlabel("layer  (emb → 0 → … → 63)", fontsize=9.8, color=MUT)
    tks = [0, 1] + list(range(8, NLp1, 8)) + [NLp1 - 1]
    ax.set_xticks(tks); ax.set_xticklabels([lab[t] for t in tks])
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.legend(fontsize=8.5, framealpha=0.95, edgecolor=GRID, loc="upper left")

fig.suptitle("Residual adds: early layers amplify, the mid-stack adds orthogonal new content (≈no "
             "shrinkage), the final layer erases — Qwen3.6-27B",
             fontsize=12, fontweight="bold", color=INK, y=0.97)
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/layer_adds.{ext}", dpi=150, bbox_inches="tight")
print("wrote layer_adds.png/.pdf")
