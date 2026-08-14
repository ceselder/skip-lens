"""Residual-stream geometry across all 64 layers (Qwen3.6-27B, mean over 353 disagreement items,
last token). Tests 'same basis throughout, only embedding/L0/last weird'.
Left: basis stability — adjacent-layer cosine of mean DIRECTIONS + how concentrated each layer's
mean direction is. Right: how much each layer writes — ‖Δ‖/‖h‖ + rotation cos(h_{L-1}, h_L).
Reads data/layer_geometry.json; writes layer_geometry.{png,pdf}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-layer-ablation")
g = json.load(open(f"{RD}/data/layer_geometry.json"))
NLp1 = g["n_hidden_states"]                 # 65: index 0 = emb, i>=1 = decoder layer i-1
xs = list(range(NLp1))                       # hidden-state index
lab = ["emb"] + [str(i) for i in range(NLp1 - 1)]
CLAY, BLUE, GRN, MUT, GRID, INK = "#D97757", "#4C72B0", "#3aa06a", "#999999", "#e6e4dd", "#333333"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.8, 5.2))
fig.subplots_adjust(top=0.83, bottom=0.14, left=0.06, right=0.945, wspace=0.22)

# adjacent_cos[i] = cos(hs[i], hs[i+1]); plot at the RIGHT endpoint i+1 (the layer just written)
adj = [None] + g["adjacent_cos"]             # align: adj[i] = cos(hs[i-1], hs[i])
conc = g["mean_dir_concentration"]
axL.plot(xs[1:], adj[1:], "-o", color=BLUE, lw=2, ms=3.5, label="adjacent-layer cosine  cos(h$_{L-1}$, h$_L$ dir)")
axL.plot(xs, conc, "-s", color=CLAY, lw=2, ms=3.5, label="mean-direction concentration  ‖mean of unit vecs‖")
axL.set_ylim(0, 1.05); axL.set_title("Basis stability across layers", fontsize=11.5, color=INK, pad=6)
axL.set_ylabel("cosine / concentration (0–1)", fontsize=9.8, color=MUT)
for x0 in (1, 63):
    axL.annotate("", xy=(x0, adj[x0]), xytext=(x0, adj[x0] - 0.14),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1))
axL.text(1.5, 0.10, "L0 overwrites\nembedding", fontsize=8, color=INK)
axL.text(56.5, 0.30, "L63 rewrites\nfor unembed", fontsize=8, color=INK, ha="center")

rel = [x if x is not None else float("nan") for x in g["add_rel"]]
axR.plot(xs, rel, "-o", color=GRN, lw=2, ms=3.5, label="relative write  ‖Δ$_L$‖ / ‖h$_L$‖")
axR.plot(xs, g["rot_cos"], "-^", color=MUT, lw=1.8, ms=3.5, label="rotation  cos(h$_{L-1}$, h$_L$)")
axR.set_title("How much each layer adds", fontsize=11.5, color=INK, pad=6)
axR.set_ylabel("relative write  /  rotation cosine", fontsize=9.8, color=MUT)

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
    ax.legend(fontsize=8.6, framealpha=0.95, edgecolor=GRID, loc="lower left")

fig.suptitle("Residual stream is one slowly-refined basis from L1–L61; the embedding, L0, and the final "
             "layer are the outliers\n(Qwen3.6-27B · mean over 353 items · directions = mean of unit vectors)",
             fontsize=12, fontweight="bold", color=INK, y=0.97)
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/layer_geometry.{ext}", dpi=150, bbox_inches="tight")
print("wrote layer_geometry.png/.pdf")
