"""Current-token vs future-only skip-lens target (both trained L62, fed raw residual).

Does adding the CURRENT token (the token at the activation's own position) to the reconstruction
target — vs the standard future-only continuation — make the mismatched skip-lens read the model's
workspace more? Sonnet-5 judge, 353 disagreement items, fed-layer depth sweep. Writes
curtok_vs_future.{png,pdf} + data/curtok_vs_future.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-layer-ablation")


def load(f):
    return json.load(open(f"{RD}/data/{f}"))["by_fed_layer"]


fut = load("judged_L62train_local.json")     # standard skip-lens: future-only target
cur = load("judged_L62curtok_local.json")    # target = current token + future
LAYERS = sorted(int(k) for k in fut)

CLAY, TEAL, INK, MUT, GRID = "#D97757", "#178A8A", "#333333", "#777777", "#e6e4dd"
ARMS = [(fut, CLAY, "o", "future-only target  (standard skip-lens)"),
        (cur, TEAL, "D", "current-token + future target")]


def ser(d, m):
    return [d[str(l)][m] for l in LAYERS], [d[str(l)][m + "_sem"] for l in LAYERS]


fig, (axW, axA) = plt.subplots(1, 2, figsize=(13, 6.0), sharey=True)
fig.subplots_adjust(top=0.72, bottom=0.11, left=0.06, right=0.985, wspace=0.07)

for ax, metric, ttl in [
    (axW, "agree_jlens", "Workspace agreement\nreadout matches the J-lens top-k concepts"),
    (axA, "agree_answer", "Surface-answer agreement\nreadout matches the model's actual next tokens"),
]:
    for d, col, mk, lab in ARMS:
        y, e = ser(d, metric)
        ax.fill_between(LAYERS, [a - b for a, b in zip(y, e)], [a + b for a, b in zip(y, e)],
                        color=col, alpha=0.12, lw=0, zorder=1)
        ax.plot(LAYERS, y, "-", marker=mk, color=col, lw=2.2, ms=6.5, label=lab, zorder=3,
                markeredgecolor="white", markeredgewidth=0.7)
    ax.axvspan(8, 38, color="#efece4", alpha=0.8, zorder=0)   # J-lens target unreliable (deep)
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
fp = max(LAYERS, key=lambda l: fut[str(l)]["agree_jlens"])
cp = max(LAYERS, key=lambda l: cur[str(l)]["agree_jlens"])
# annotate L62 (native, the true peak now) — future above, current-token below
axW.annotate(f"{fut['62']['agree_jlens']:.2f}", (62, fut['62']['agree_jlens']),
             textcoords="offset points", xytext=(-4, 8), color=CLAY, fontsize=9.5, fontweight="bold", ha="right")
axW.annotate(f"{cur['62']['agree_jlens']:.2f}", (62, cur['62']['agree_jlens']),
             textcoords="offset points", xytext=(-4, -16), color=TEAL, fontsize=9.5, fontweight="bold", ha="right")
axW.text(23, 0.045, "J-lens target\nunreliable here", fontsize=8.2, color=MUT, ha="center", style="italic")
axW.legend(loc="upper left", fontsize=9.2, framealpha=0.96, edgecolor=GRID)

fig.suptitle(
    "Scored FAIRLY (each layer vs its OWN J-lens workspace), the current-token target does NOT beat\n"
    "future-only — it is ≤ future-only across the trustworthy range (L42–L62); the earlier deeper-peak\n"
    "was an artifact of scoring every layer against L42’s workspace",
    fontsize=12.5, fontweight="bold", color=INK, y=0.995, linespacing=1.4)
fig.text(0.5, 0.87,
         "Qwen3.6-27B · both train L62, fed raw · scored vs each layer’s LOCAL J-lens workspace · "
         "353 items · shaded = J-lens linearization unreliable (deep) · dotted = fed L42",
         fontsize=9.3, color=MUT, ha="center")

for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/curtok_vs_future.{ext}", dpi=150, bbox_inches="tight")

out = {"layers": LAYERS, "n_items": 353,
       "future_only": {l: fut[str(l)] for l in LAYERS},
       "current_token": {l: cur[str(l)] for l in LAYERS},
       "future_peak": {"layer": fp, "agree_jlens": fut[str(fp)]["agree_jlens"]},
       "current_peak": {"layer": cp, "agree_jlens": cur[str(cp)]["agree_jlens"]}}
json.dump(out, open(f"{RD}/data/curtok_vs_future.json", "w"), indent=1)
print(f"wrote curtok_vs_future.png/.pdf | future peak L{fp}={fut[str(fp)]['agree_jlens']:.3f} | "
      f"current-token peak L{cp}={cur[str(cp)]['agree_jlens']:.3f}")
PY = None
