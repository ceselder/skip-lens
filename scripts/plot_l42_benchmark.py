"""L42-workspace benchmark, chat-native, per eval category.

For each category, how well does feeding each depth's residual let the lens recover the INTERESTING
L42 workspace (the intermediate, answer-distinct concept) — scored full-readout, concept-anywhere,
against the fixed L42 J-lens top-k. The skip-lens thesis = feed-L42 (or L48) beats feed-L62 where the
intermediate workspace is a distinct abstract concept. Reads judged_chat500_L42tgt.json (detail).
Writes l42_benchmark.{png,pdf} + data/l42_benchmark.json.
"""
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RD = os.path.expanduser("~/shared/reports/skiplens-layer-ablation")
det = json.load(open(f"{RD}/data/judged_chat500_L42tgt.json"))["detail"]

FEDS = [62, 48, 42, 34]
CATS = ["association", "multilingual", "order_of_ops", "multihop", "poetry", "typo"]
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in det:
    c = r["name"].split("[")[0]
    if r.get("agree_jlens") is not None:
        agg[c][r["fed_layer"]].append(r["agree_jlens"] / 2.0)   # 0/1/2 -> 0-1

mean = {c: {l: (np.mean(agg[c][l]) if agg[c][l] else np.nan) for l in FEDS} for c in CATS}
sem = {c: {l: (np.std(agg[c][l]) / max(1, np.sqrt(len(agg[c][l]))) if agg[c][l] else 0) for l in FEDS} for c in CATS}

COL = {62: "#4C72B0", 48: "#3aa06a", 42: "#D97757", 34: "#b9b3a7"}
LAB = {62: "feed L62 (penultimate = answer)", 48: "feed L48", 42: "feed L42 (skip = intermediate)", 34: "feed L34 (deep)"}
INK, MUT, GRID = "#333333", "#777777", "#e6e4dd"

fig, ax = plt.subplots(figsize=(12.5, 5.6))
fig.subplots_adjust(top=0.83, bottom=0.12, left=0.07, right=0.985)
x = np.arange(len(CATS)); w = 0.2
for i, l in enumerate(FEDS):
    ys = [mean[c][l] for c in CATS]
    es = [sem[c][l] for c in CATS]
    ax.bar(x + (i - 1.5) * w, ys, w, yerr=es, capsize=2, color=COL[l], label=LAB[l],
           edgecolor="white", linewidth=0.6, error_kw=dict(lw=1, ecolor=MUT))
ax.set_xticks(x); ax.set_xticklabels(CATS, fontsize=9.5)
ax.set_ylabel("agreement with the L42 workspace (0–1)", fontsize=10, color=MUT)
ax.set_ylim(0, 1.05)
ax.grid(True, axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUT)
ax.legend(fontsize=8.8, framealpha=0.95, edgecolor=GRID, ncol=2, loc="upper right")
fig.suptitle("Skip-lens recovers the intermediate (L42) workspace best where it is a distinct abstract "
             "concept — chat-native, per category\n"
             "(feed each depth → score full readout vs the L42 J-lens top-k, concept-anywhere · "
             "Qwen3.6-27B · 353 items)", fontsize=11.5, fontweight="bold", color=INK, y=0.98)
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/l42_benchmark.{ext}", dpi=150, bbox_inches="tight")

out = {"categories": CATS, "fed_layers": FEDS, "metric": "agree_with_L42_workspace_0to1",
       "n_per_cat": {c: len(agg[c][42]) // 1 for c in CATS},
       "mean": {c: {str(l): (None if np.isnan(mean[c][l]) else round(float(mean[c][l]), 3)) for l in FEDS} for c in CATS},
       "sem": {c: {str(l): round(float(sem[c][l]), 3) for l in FEDS} for c in CATS}}
json.dump(out, open(f"{RD}/data/l42_benchmark.json", "w"), indent=1)
print("wrote l42_benchmark.png/.pdf")
for c in CATS:
    print(f"  {c:14s} " + " ".join(f"L{l}={mean[c][l]:.2f}" for l in FEDS))
