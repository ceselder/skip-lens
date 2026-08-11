"""L62-trained vs L42-trained skip-lens, 8-token readouts — one panel per metric.

workspace  = readout matches the layer's own J-lens top-k concepts   (Sonnet, 0-1)
answer     = readout matches the model's actual next tokens          (Sonnet, 0-1)
coherence  = readout is fluent/coherent English                      (Sonnet, 0-1)
logprob    = base model's mean per-token logprob of the readout's first 3 tokens as a
             real continuation of the context (judge-free; higher = more literal answer)

Both fed the raw residual at each depth. Writes eightok_4panel.{png,pdf} + data/eightok_4panel.json.
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


WS = {"L62": load("judged_L62train_local_8tok.json"), "L42": load("judged_L42train_local_8tok.json")}
CO = {"L62": load("coherence_L62train_8tok.json"), "L42": load("coherence_L42train_8tok.json")}
LP = {"L62": load("fed_L62train_logprob.json"), "L42": load("fed_L42train_logprob.json")}
LPREF = {}
for k, f in [("L62", "fed_L62train_logprob.json"), ("L42", "fed_L42train_logprob.json")]:
    p = f"{RD}/data/{f}"
    LPREF[k] = json.load(open(p)).get("actual_reference_logprob") if os.path.exists(p) else None

LAYERS = sorted(int(k) for k in WS["L62"])
CLAY, BLUE, INK, MUT, GRID = "#D97757", "#4C72B0", "#333333", "#777777", "#e6e4dd"
STY = {"L62": (CLAY, "o", "skip-lens (futurelens on L62)"), "L42": (BLUE, "s", "control (futurelens on L42)")}

fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
fig.subplots_adjust(top=0.90, bottom=0.07, left=0.07, right=0.985, hspace=0.30, wspace=0.13)
(axW, axA), (axC, axL) = axes


def plot_panel(ax, src, key, title, ylabel, ylim, shade_deep=False, sem_key=None):
    if shade_deep:
        ax.axvspan(8, 38, color="#efece4", alpha=0.8, zorder=0)
    for m, (col, mk, lab) in STY.items():
        d = src.get(m)
        if d is None:
            continue
        ys = [d[str(l)].get(key) for l in LAYERS]
        xs = [l for l, y in zip(LAYERS, ys) if y is not None]
        yv = [y for y in ys if y is not None]
        if sem_key:
            e = [d[str(l)].get(sem_key, 0) for l in LAYERS if d[str(l)].get(key) is not None]
            ax.fill_between(xs, [a - b for a, b in zip(yv, e)], [a + b for a, b in zip(yv, e)],
                            color=col, alpha=0.12, lw=0, zorder=1)
        ax.plot(xs, yv, "-", marker=mk, color=col, lw=2.1, ms=6, label=lab, zorder=3,
                markeredgecolor="white", markeredgewidth=0.7)
    ax.axvline(42, color=MUT, ls=(0, (2, 2)), lw=1.1, zorder=2)
    ax.set_title(title, fontsize=11.5, color=INK, pad=7)
    ax.set_xlabel("fed residual layer  (shallow → penultimate)", fontsize=9.8, color=MUT)
    ax.set_ylabel(ylabel, fontsize=9.8, color=MUT)
    ax.set_xticks(LAYERS)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUT, labelsize=9)
    ax.legend(fontsize=8.8, framealpha=0.95, edgecolor=GRID)


plot_panel(axW, WS, "agree_jlens", "Workspace agreement (own-layer J-lens)", "Sonnet-5 judge score (0–1)", (0, 0.8),
           shade_deep=True, sem_key="agree_jlens_sem")
plot_panel(axA, WS, "agree_answer", "Surface-answer agreement", "Sonnet-5 judge score (0–1)", (0, 0.7),
           sem_key="agree_answer_sem")
plot_panel(axC, CO, "coherence", "Coherence (fluency of readout)", "Sonnet-5 judge score (0–1)", (0, 0.7),
           sem_key="coherence_sem")
if LP["L62"] is not None and LP["L42"] is not None:
    plot_panel(axL, LP, "mean_logprob", "Logprob of readout as a real continuation (judge-free)",
               "continuation average logprob", None, sem_key="sem")
    _cp = f"{RD}/data/canonical_ref.json"
    if os.path.exists(_cp):
        cref = json.load(open(_cp))["canonical_greedy_ref"]
        axL.axhline(cref, color=MUT, ls=":", lw=1.4,
                    label=f"model's own greedy continuation (ceiling {cref:.2f})")
    else:
        for m, (col, _, _) in STY.items():
            if LPREF.get(m) is not None:
                axL.axhline(LPREF[m], color=col, ls=":", lw=1.3,
                            label=f"{m}: actual-continuation ref {LPREF[m]:.2f}")
    axL.legend(fontsize=8.2, framealpha=0.95, edgecolor=GRID)
else:
    axL.text(0.5, 0.5, "logprob computing…", ha="center", va="center", fontsize=13, color=MUT)
    axL.set_title("Logprob of readout as a real continuation (judge-free)", fontsize=11.5, color=INK)
    axL.axis("off")

fig.suptitle("Skip-lens L62-trained vs L42-trained · 8-token readouts · fed raw residual at each depth "
             "(Qwen3.6-27B, 353 disagreement items)",
             fontsize=13, fontweight="bold", color=INK, y=0.975)

for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/eightok_4panel.{ext}", dpi=150, bbox_inches="tight")

out = {"layers": LAYERS, "n_items": 353, "readout_tokens": 8,
       "workspace": {m: {l: WS[m][str(l)]["agree_jlens"] for l in LAYERS} for m in WS},
       "answer": {m: {l: WS[m][str(l)]["agree_answer"] for l in LAYERS} for m in WS},
       "coherence": {m: {l: CO[m][str(l)]["coherence"] for l in LAYERS} for m in CO if CO[m]},
       "logprob": {m: ({l: LP[m][str(l)]["mean_logprob"] for l in LAYERS if str(l) in LP[m]}
                       if LP[m] else None) for m in LP},
       "logprob_actual_ref": LPREF}
json.dump(out, open(f"{RD}/data/eightok_4panel.json", "w"), indent=1)
print("wrote eightok_4panel.png/.pdf | logprob:", {m: (LP[m] is not None) for m in LP})
