"""Layer x layer cosine-similarity heatmap of the residual stream in Qwen3.6-27B.
Left: raw mean-direction cosine. Right: rogue-dim-corrected (per-dim standardized). The bright
middle block = one shared, slowly-drifting basis (L1-L61); the embedding row/col and the final
layer stand apart. Reads data/layer_cos_matrix.json; writes layer_cos_matrix.{png,pdf}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RD = os.path.expanduser("~/shared/reports/skiplens-layer-ablation")
g = json.load(open(f"{RD}/data/layer_cos_matrix.json"))
NLp1 = g["n_hidden_states"]
lab = ["emb"] + [str(i) for i in range(NLp1 - 1)]
INK, MUT, GRID = "#333333", "#777777", "#e6e4dd"

fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.2))
fig.subplots_adjust(top=0.82, bottom=0.11, left=0.055, right=0.965, wspace=0.22)
for ax, key, ttl in [(axes[0], "cos_raw", "Raw cosine of mean directions"),
                     (axes[1], "cos_standardized", "Rogue-dim corrected (per-dim standardized)")]:
    M = np.array(g[key])
    im = ax.imshow(M, cmap="magma", vmin=0.0, vmax=1.0, origin="upper", aspect="equal")
    ax.set_title(ttl, fontsize=11.5, color=INK, pad=7)
    tks = [0, 1] + list(range(8, NLp1, 8)) + [NLp1 - 1]
    ax.set_xticks(tks); ax.set_xticklabels([lab[t] for t in tks], fontsize=8)
    ax.set_yticks(tks); ax.set_yticklabels([lab[t] for t in tks], fontsize=8)
    ax.set_xlabel("layer", fontsize=9.6, color=MUT); ax.set_ylabel("layer", fontsize=9.6, color=MUT)
    ax.tick_params(colors=MUT)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("cosine similarity", fontsize=9, color=MUT)
    cb.ax.tick_params(colors=MUT, labelsize=8)

fig.suptitle("Residual-stream layer×layer cosine similarity — Qwen3.6-27B\n"
             "one bright shared-basis block across L1–L61; the embedding and the final layer stand apart "
             "(353 items · mean of unit vectors)",
             fontsize=12.5, fontweight="bold", color=INK, y=0.97)
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/layer_cos_matrix.{ext}", dpi=150, bbox_inches="tight")
print("wrote layer_cos_matrix.png/.pdf")
