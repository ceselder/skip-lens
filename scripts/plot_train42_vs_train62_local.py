"""Trained-on-L42 vs trained-on-L62 skip-lens, scored against each layer's OWN J-lens workspace.

Both fed the raw residual at each depth; workspace target = topk(lens(J_{l->62}.h_l)) per fed layer
(the fair, local version — not the fixed L42 reference). Shows whether the penultimate-trained lens
reads the local workspace more than the intermediate-trained one. Writes train42_vs_train62_local.{png,pdf}
+ data/train42_vs_train62_local.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-layer-ablation")


def load(f):
    return json.load(open(f"{RD}/data/{f}"))["by_fed_layer"]


l62 = load("judged_L62train_local.json")   # trained on L62 (penultimate) = mismatch / standard skip-lens
l42 = load("judged_L42train_local.json")   # trained on L42 (intermediate) = matched
LAYERS = sorted(int(k) for k in l62)

CLAY, BLUE, INK, MUT, GRID = "#D97757", "#4C72B0", "#333333", "#777777", "#e6e4dd"
ARMS = [(l62, CLAY, "o", "trained L62  (penultimate / standard skip-lens)"),
        (l42, BLUE, "s", "trained L42  (intermediate)")]


def ser(d, m):
    return [d[str(l)][m] for l in LAYERS], [d[str(l)][m + "_sem"] for l in LAYERS]


fig, (axW, axA) = plt.subplots(1, 2, figsize=(13, 6.0), sharey=True)
fig.subplots_adjust(top=0.72, bottom=0.11, left=0.06, right=0.985, wspace=0.07)

for ax, metric, ttl in [
    (axW, "agree_jlens", "Workspace agreement\nreadout matches the layer's OWN J-lens top-k concepts"),
    (axA, "agree_answer", "Surface-answer agreement\nreadout matches the model's actual next tokens"),
]:
    ax.axvspan(8, 38, color="#efece4", alpha=0.8, zorder=0)          # J-lens target unreliable (deep)
    for d, col, mk, lab in ARMS:
        y, e = ser(d, metric)
        ax.fill_between(LAYERS, [a - b for a, b in zip(y, e)], [a + b for a, b in zip(y, e)],
                        color=col, alpha=0.12, lw=0, zorder=1)
        ax.plot(LAYERS, y, "-", marker=mk, color=col, lw=2.2, ms=6.5, label=lab, zorder=3,
                markeredgecolor="white", markeredgewidth=0.7)
    ax.axvline(42, color=MUT, ls=(0, (2, 2)), lw=1.1, zorder=2)
    ax.set_title(ttl, fontsize=11.5, color=INK, pad=9, linespacing=1.35)
    ax.set_xlabel("fed residual layer   (shallow → penultimate)", fontsize=10.5, color=MUT)
    ax.set_xticks(LAYERS)
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9.5)

axW.set_ylabel("agreement  (Sonnet-5 judge, 0–1)", fontsize=10.5, color=MUT)
axW.set_ylim(0.0, 0.82)
for lay, dy62, dy42 in [(62, 8, -16), (42, 8, -16)]:
    axW.annotate(f"{l62[str(lay)]['agree_jlens']:.2f}", (lay, l62[str(lay)]['agree_jlens']),
                 textcoords="offset points", xytext=(-3, dy62), color=CLAY, fontsize=9, fontweight="bold", ha="right")
    axW.annotate(f"{l42[str(lay)]['agree_jlens']:.2f}", (lay, l42[str(lay)]['agree_jlens']),
                 textcoords="offset points", xytext=(-3, dy42), color=BLUE, fontsize=9, fontweight="bold", ha="right")
axW.text(23, 0.045, "J-lens target\nunreliable here", fontsize=8.2, color=MUT, ha="center", style="italic")
axW.legend(loc="upper left", fontsize=9.2, framealpha=0.96, edgecolor=GRID)

fig.suptitle(
    "Scored against each layer's OWN workspace: the penultimate-trained (L62) skip-lens reads the local\n"
    "workspace MORE than the intermediate-trained (L42) one across L42–L62 (they cross at L34; both\n"
    "collapse where the J-lens linearization is unreliable)",
    fontsize=12.4, fontweight="bold", color=INK, y=0.995, linespacing=1.4)
fig.text(0.5, 0.86,
         "Qwen3.6-27B · 150k token-matched FineWeb pair (only source layer differs) · fed raw residual · "
         "local per-layer J-lens target · 353 items · shaded = deep (J-lens unreliable) · dotted = fed L42",
         fontsize=9.3, color=MUT, ha="center")

for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/train42_vs_train62_local.{ext}", dpi=150, bbox_inches="tight")

out = {"layers": LAYERS, "n_items": 353, "scoring": "local per-layer J-lens workspace",
       "trained_L62": {l: l62[str(l)] for l in LAYERS},
       "trained_L42": {l: l42[str(l)] for l in LAYERS}}
json.dump(out, open(f"{RD}/data/train42_vs_train62_local.json", "w"), indent=1)
print("wrote train42_vs_train62_local.png/.pdf")
for l in LAYERS:
    print(f"  L{l}: L62-trained {l62[str(l)]['agree_jlens']:.3f}  L42-trained {l42[str(l)]['agree_jlens']:.3f}")
