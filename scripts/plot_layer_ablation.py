"""Source-layer ablation for the skip-lens decoder on Qwen3.6-27B — RAW + Jacobian arms.

Two decoders trained on an IDENTICAL 150k token-matched FineWeb corpus; the only difference is the
source layer injected during training (L62 = penultimate, the standard skip-lens, vs L42 = an
intermediate). At test we feed each the activation from a range of depths, either RAW or mapped
through the fitted J-lens Jacobian J_{l->62} into the penultimate basis, and a Sonnet-5 judge scores
each readout: agreement with the J-lens top-k concepts (the model's hidden "workspace") vs the
model's actual next tokens (the surface answer), on 353 disagreement items.

Claim: penultimate-training reads the workspace more than L42-training; mapping the fed activation
into the penultimate basis (the Jacobian) sharpens the workspace read for both, and the standard
skip-lens (train L62) fed J.L42 reads the workspace best.

Writes layer_ablation.{png,pdf} + data/layer_ablation.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-layer-ablation")


def load(f):
    return json.load(open(f"{RD}/data/{f}"))["by_fed_layer"]


D = {
    "L62_raw": load("judged_L62train.json"),
    "L62_jac": load("judgedjac_L62train.json"),
    "L42_raw": load("judged_L42train.json"),
    "L42_jac": load("judgedjac_L42train.json"),
}
CLAY, BLUE, INK, MUT, GRID = "#D97757", "#4C72B0", "#333333", "#777777", "#e6e4dd"
# (key, color, linestyle, marker, label)
ARMS = [
    ("L62_raw", CLAY, "-",  "o", "train L62 · raw L42   (mismatch / skip-lens)"),
    ("L62_jac", CLAY, "--", "s", "train L62 · J·L42      (mismatch + Jacobian)"),
    ("L42_raw", BLUE, "-",  "o", "train L42 · raw L42   (matched)"),
    ("L42_jac", BLUE, "--", "s", "train L42 · J·L42      (matched + Jacobian)"),
]


def xy(d, metric):
    layers = sorted((int(k) for k in d), reverse=False)
    return layers, [d[str(l)][metric] for l in layers]


fig, (axW, axA) = plt.subplots(1, 2, figsize=(13.5, 6.2), sharey=True)
fig.subplots_adjust(top=0.72, bottom=0.11, left=0.055, right=0.985, wspace=0.07)

for ax, metric, ttl in [
    (axW, "agree_jlens", "Workspace agreement\nreadout matches the J-lens top-k concepts"),
    (axA, "agree_answer", "Surface-answer agreement\nreadout matches the model's actual next tokens"),
]:
    for key, col, ls, mk, lab in ARMS:
        x, y = xy(D[key], metric)
        ax.plot(x, y, ls=ls, marker=mk, color=col, lw=2.1, ms=6, label=lab, zorder=3,
                markeredgecolor="white", markeredgewidth=0.7, alpha=0.95)
    ax.axvline(42, color=MUT, ls=(0, (2, 2)), lw=1.2, zorder=1)
    ax.set_title(ttl, fontsize=11.5, color=INK, pad=9, linespacing=1.35)
    ax.set_xlabel("fed residual layer   (shallow → penultimate)", fontsize=10.5, color=MUT)
    ax.set_xticks([10, 18, 26, 34, 42, 48, 55, 62])
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9.5)

axW.set_ylabel("agreement  (Sonnet-5 judge, 0–1)", fontsize=10.5, color=MUT)
axW.set_ylim(0.0, 0.75)

# annotate the four fed-L42 workspace values (the headline point)
for key, col, ls, mk, lab in ARMS:
    v = D[key]["42"]["agree_jlens"]
    dy = {"L62_jac": 10, "L42_jac": -2, "L62_raw": -14, "L42_raw": -26}[key]
    axW.annotate(f"{v:.2f}", (42, v), textcoords="offset points", xytext=(7, dy),
                 color=col, fontsize=9, fontweight="bold")

axW.legend(loc="lower center", fontsize=8.6, framealpha=0.96, edgecolor=GRID, ncol=1)

fig.suptitle(
    "Penultimate-trained lens reads the workspace more than an L42-trained one — and mapping the fed\n"
    "activation into the penultimate basis (the fitted J-lens Jacobian) sharpens it further for both",
    fontsize=12.6, fontweight="bold", color=INK, y=0.995, linespacing=1.4)
fig.text(0.5, 0.87,
         "Qwen3.6-27B · 150k token-matched FineWeb decoder pair (only the source layer differs) · "
         "raw vs J·h_l fed at each depth · 353 disagreement items · dotted line = fed L42",
         fontsize=9.3, color=MUT, ha="center")

for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/layer_ablation.{ext}", dpi=150, bbox_inches="tight")

layers_all = sorted((int(k) for k in D["L62_raw"]))
out = {"layers_raw": layers_all, "layers_jac": sorted((int(k) for k in D["L62_jac"])), "n_items": 353,
       "arms": {k: {l: D[k][str(l)] for l in sorted((int(x) for x in D[k]))} for k in D},
       "fed_L42": {k: {"agree_jlens": D[k]["42"]["agree_jlens"], "agree_answer": D[k]["42"]["agree_answer"]} for k in D}}
json.dump(out, open(f"{RD}/data/layer_ablation.json", "w"), indent=1)
print("wrote layer_ablation.png/.pdf + data/layer_ablation.json")
for k in D:
    print(f"  fed L42 {k}: jlens {D[k]['42']['agree_jlens']:.3f}  answer {D[k]['42']['agree_answer']:.3f}")
