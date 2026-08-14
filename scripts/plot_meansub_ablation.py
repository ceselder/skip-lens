"""Mean-subtraction ablation: does injecting the DEVIATION-from-mean (train L62-mean-subtracted,
test L42-mean-subtracted) reduce the train/test layer mismatch vs injecting the raw activation?

Two 500-pair lenses, identical except the injected representation, fed the residual at each depth
on the disagreement set; each fed layer scored vs its OWN local J-lens workspace. Faint reference =
the full raw skip-lens (150k pairs). Panels: workspace agreement | surface-answer agreement.
Writes meansub_ablation.{png,pdf} + data/meansub_ablation.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-layer-ablation")


def load(f):
    p = f"{RD}/data/{f}"
    return json.load(open(p))["by_fed_layer"] if os.path.exists(p) else None


NORMED = load("judged_meansubnorm500_8tok.json")   # mean-sub in NORMALIZED space (the right way)
MEANSUB = load("judged_meansub500_8tok.json")       # mean-sub with raw magnitude (the cursed way)
RAW = load("judged_raw500_8tok.json")
REF = load("judged_L62train_local_8tok.json")  # full raw skip-lens, 150k pairs (context)

LAYERS = sorted(int(k) for k in RAW)
CLAY, LCLAY, BLUE, MUT, GRID, INK = "#D97757", "#E8A87C", "#4C72B0", "#999999", "#e6e4dd", "#333333"
SERIES = [
    ("NORMED", NORMED, CLAY, "o", "-", "mean-subtracted in NORMALIZED space (inject ĥ−mean_dir)"),
    ("RAWMEAN", MEANSUB, LCLAY, "D", "--", "mean-subtracted, raw magnitude (the cursed version)"),
    ("RAW", RAW, BLUE, "s", "-", "raw control (no subtraction)"),
    ("REF", REF, MUT, "^", ":", "reference: raw skip-lens, 150k pairs"),
]

fig, (axW, axA) = plt.subplots(1, 2, figsize=(13.5, 5.4))
fig.subplots_adjust(top=0.84, bottom=0.13, left=0.07, right=0.985, wspace=0.16)


def panel(ax, key, sem_key, title):
    for _, src, col, mk, ls, lab in SERIES:
        if src is None:
            continue
        xs = [l for l in LAYERS if str(l) in src and src[str(l)].get(key) is not None]
        yv = [src[str(l)][key] for l in xs]
        e = [src[str(l)].get(sem_key, 0) for l in xs]
        ax.fill_between(xs, [a - b for a, b in zip(yv, e)], [a + b for a, b in zip(yv, e)],
                        color=col, alpha=0.10, lw=0, zorder=1)
        ax.plot(xs, yv, ls, marker=mk, color=col, lw=2.1, ms=6, label=lab, zorder=3,
                markeredgecolor="white", markeredgewidth=0.7)
    ax.axvline(42, color=MUT, ls=(0, (2, 2)), lw=1.1, zorder=2)
    ax.set_title(title, fontsize=11.5, color=INK, pad=7)
    ax.set_xlabel("fed residual layer  (shallow → penultimate)", fontsize=9.8, color=MUT)
    ax.set_ylabel("Sonnet-5 judge score (0–1)", fontsize=9.8, color=MUT)
    ax.set_xticks(LAYERS)
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.legend(fontsize=8.6, framealpha=0.95, edgecolor=GRID)


panel(axW, "agree_jlens", "agree_jlens_sem", "Workspace agreement (own-layer J-lens)")
panel(axA, "agree_answer", "agree_answer_sem", "Surface-answer agreement")

fig.suptitle("Mean-subtraction works — but only in NORMALIZED space: +0.12 workspace @ L42 (≈6 SEM), "
             "nearly matching the 150k lens; raw-magnitude centering was within noise\n"
             "(Qwen3.6-27B · 500-pair lenses vs 150k reference · 8-token readouts · 353 disagreement items)",
             fontsize=11.5, fontweight="bold", color=INK, y=0.985)

for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/meansub_ablation.{ext}", dpi=150, bbox_inches="tight")

out = {"layers": LAYERS, "n_items": 353, "readout_tokens": 8,
       "workspace": {n: ({l: s[str(l)]["agree_jlens"] for l in LAYERS if str(l) in s} if s else None)
                     for n, s, *_ in SERIES},
       "answer": {n: ({l: s[str(l)]["agree_answer"] for l in LAYERS if str(l) in s} if s else None)
                  for n, s, *_ in SERIES}}
json.dump(out, open(f"{RD}/data/meansub_ablation.json", "w"), indent=1)
print("wrote meansub_ablation.png/.pdf")
for n, s, *_ in SERIES:
    if s:
        print(n, {l: (round(s[str(l)]["agree_jlens"], 3) if str(l) in s else None) for l in LAYERS})
