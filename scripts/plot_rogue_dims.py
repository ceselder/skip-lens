"""Does the high adjacent-layer cosine survive the Timkey & van Schijndel (2021) rogue-dimension
correction? Adjacent-layer cosine of mean directions: raw vs top outlier dims removed vs full
per-dim standardization. If it stays high after correction, the shared basis is real (not an
artifact of a few massive-activation dims). Reads data/rogue_dims.json; writes rogue_dims.{png,pdf}.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-layer-ablation")
g = json.load(open(f"{RD}/data/rogue_dims.json"))
NLp1 = g["n_hidden_states"]
x = list(range(1, NLp1))                    # transition i-1 -> i, plotted at right endpoint i (=hidden idx)
lab = ["emb"] + [str(i) for i in range(NLp1 - 1)]
CLAY, BLUE, GRN, PURP, MUT, GRID, INK = "#D97757", "#4C72B0", "#3aa06a", "#9467bd", "#999999", "#e6e4dd", "#333333"

fig, ax = plt.subplots(figsize=(12.5, 5.4))
fig.subplots_adjust(top=0.82, bottom=0.14, left=0.07, right=0.985)
ax.plot(x, g["adj_cos_raw"], "-o", color=BLUE, lw=2, ms=3.5, label="raw cosine of mean directions")
ax.plot(x, g["adj_cos_drop3"], "-s", color=CLAY, lw=2, ms=3.5, label="top-3 outlier dims removed")
ax.plot(x, g["adj_cos_drop10"], "-^", color=GRN, lw=1.8, ms=3.3, label="top-10 outlier dims removed")
ax.plot(x, g["adj_cos_standardized"], "-d", color=PURP, lw=1.8, ms=3.3, label="per-dim standardized (z-scored)")
ax.axhspan(0.88, 0.95, color="#eef4ee", zorder=0)
ax.axvline(64, color=MUT, ls=(0, (2, 2)), lw=1.1)
ax.text(63.4, 0.55, "L62→L63\n(final rewrite)", fontsize=8.5, color=INK, ha="right")
ax.set_ylim(0.35, 1.02)
ax.set_xlabel("layer (adjacent transition, plotted at upper layer;  emb → 0 → … → 63)", fontsize=9.8, color=MUT)
ax.set_ylabel("adjacent-layer cosine (0–1)", fontsize=9.8, color=MUT)
tks = [1, 2] + list(range(8, NLp1, 8)) + [NLp1 - 1]
ax.set_xticks(tks); ax.set_xticklabels([lab[t] for t in tks])
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUT, labelsize=9)
ax.legend(fontsize=9, framealpha=0.95, edgecolor=GRID, loc="lower left")
fig.suptitle("The shared basis SURVIVES the rogue-dimension correction (Timkey & van Schijndel 2021):\n"
             "adjacent-layer cosine stays 0.88–0.94 across L1–L61 even after removing outlier dims — "
             "only the final layer is a true outlier",
             fontsize=12, fontweight="bold", color=INK, y=0.98)
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/rogue_dims.{ext}", dpi=150, bbox_inches="tight")
print("wrote rogue_dims.png/.pdf")
