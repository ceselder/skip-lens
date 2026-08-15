"""Scale test: the mean-centering +0.119 @ L42 was a small-data artifact. At 5k pairs raw catches up
(gap -0.003); mean-centering caps out. Data (500->5k->150k for raw) is the real lever. Writes
meansub_scale.{png,pdf} + data/meansub_scale.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-layer-ablation")
raw = {500: 0.411, 5000: 0.511, 150000: 0.581}
mc = {500: 0.530, 5000: 0.508}
CLAY, BLUE, MUT, GRID, INK = "#D97757", "#4C72B0", "#999999", "#e6e4dd", "#333333"

fig, ax = plt.subplots(figsize=(8.4, 5.2))
fig.subplots_adjust(top=0.86, bottom=0.13, left=0.11, right=0.96)
ax.plot(list(raw), list(raw.values()), "-s", color=BLUE, lw=2.2, ms=7, label="raw injection")
ax.plot(list(mc), list(mc.values()), "-o", color=CLAY, lw=2.2, ms=7, label="normalized mean-centering")
ax.set_xscale("log")
ax.set_xticks([500, 5000, 150000]); ax.set_xticklabels(["500", "5k", "150k"])
ax.set_xlabel("training pairs", fontsize=10, color=MUT)
ax.set_ylabel("workspace agreement @ fed L42 (0–1)", fontsize=10, color=MUT)
ax.set_ylim(0.38, 0.62)
ax.annotate("+0.119\n(small-data rescue)", xy=(500, 0.47), fontsize=8.5, color=INK, ha="center")
ax.annotate("gap → −0.003\n(raw catches up)", xy=(5000, 0.53), fontsize=8.5, color=INK, ha="center")
ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUT, labelsize=9)
ax.legend(fontsize=9.5, framealpha=0.95, edgecolor=GRID, loc="lower right")
fig.suptitle("The mean-centering win was a small-data artifact: raw catches up at 5k\n"
             "(Qwen3.6-27B · fed L42 workspace agreement · raw substrate)",
             fontsize=11.5, fontweight="bold", color=INK, y=0.97)
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/meansub_scale.{ext}", dpi=150, bbox_inches="tight")
json.dump({"raw": {str(k): v for k, v in raw.items()}, "mean_centering": {str(k): v for k, v in mc.items()},
           "metric": "workspace_agreement_fed_L42"}, open(f"{RD}/data/meansub_scale.json", "w"), indent=1)
print("wrote meansub_scale.png/.pdf")
