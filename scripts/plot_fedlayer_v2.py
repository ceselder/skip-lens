import json, math, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

R = "/home/celeste/multi-token-jlens-nla-lastlayer/results"
CK = [("joracle", "claude-joracle", "-",  "o"),
      ("naive",   "naive futurelens", "--", "s"),
      ("rlpolicy","NLA-futurelens",  ":",  "^")]
D = {c: json.load(open(f"{R}/fedlayer_{c}_plot.json")) for c, _, _, _ in CK}
layers = D["naive"]["fed_layers"]
pct = [16, 28, 41, 53, 66, 78, 91, 97, 98]
xt = list(range(len(layers)))
xl = [f"L{l}\n{p}%" for l, p in zip(layers, pct)]

# colour = SERIES (identical top & bottom)
SER = [("agree_jlens_raw", "#C44E52", r"raw $h_\ell\!\to$ J-lens (workspace)"),
       ("agree_answer_raw", "#4C72B0", r"raw $h_\ell\!\to$ answer (continuation)"),
       ("agree_jlens_jac",  "#8172B3", r"$J_{\ell\to62}h_\ell\!\to$ J-lens (Jacobian)")]

def ci(vals, n):  # binomial 95% half-width; None-safe
    return [1.96 * math.sqrt(p * (1 - p) / n) if p is not None else None for p in vals]

def clean(vals, cis):
    xs = [i for i, v in enumerate(vals) if v is not None]
    return xs, [vals[i] for i in xs], [cis[i] for i in xs]

def build(ci_style):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3})
    fig = plt.figure(figsize=(14, 9))
    # 6-col grid: top panels 2 cols each; bottom centred over cols 1..5 (=> ~67% width, less stretchy)
    gs = GridSpec(2, 6, height_ratios=[1, 1.25], hspace=.42, wspace=.55)
    TOPPOS = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6]]

    for i, (c, title, _, _) in enumerate(CK):
        ax = fig.add_subplot(TOPPOS[i]); d = D[c]; n = d["n"]
        for key, col, _lab in SER:                      # top: differ by COLOUR only, same style (solid 'o')
            x, y, e = clean(d[key], ci(d[key], n))
            if ci_style == "band":
                ax.plot(x, y, "-o", color=col, ms=5)
                ax.fill_between(x, [a - b for a, b in zip(y, e)], [a + b for a, b in zip(y, e)],
                                color=col, alpha=.16, lw=0)
            else:
                ax.errorbar(x, y, yerr=e, fmt="-o", color=col, ms=5, capsize=2, lw=1.5)
        ax.set_title(f"{title}  (n={n})", fontsize=12, fontweight="bold")
        ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.5); ax.set_ylim(0, 1)
        if i == 0: ax.set_ylabel("Sonnet-5 agreement (0–1)")
    # one shared series legend on the top-left panel
    fig.axes[0].legend(handles=[Line2D([0], [0], color=col, marker="o", label=lab) for _, col, lab in SER],
                       fontsize=7, loc="lower center")

    # bottom overlay: colour = series (workspace red, answer blue), style = checkpoint
    axo = fig.add_subplot(gs[1, 1:5])
    for c, title, ls, mk in CK:
        d = D[c]; n = d["n"]
        for key, col, _lab in SER[:2]:                  # workspace + answer (the inverted contrast)
            x, y, e = clean(d[key], ci(d[key], n))
            if ci_style == "band":
                axo.plot(x, y, ls, color=col, marker=mk, ms=5, lw=1.8)
                axo.fill_between(x, [a - b for a, b in zip(y, e)], [a + b for a, b in zip(y, e)],
                                 color=col, alpha=.10, lw=0)
            else:
                axo.errorbar(x, y, yerr=e, fmt=ls, color=col, marker=mk, ms=5, capsize=2, lw=1.6)
    axo.set_xticks(xt); axo.set_xticklabels(xl, fontsize=8); axo.set_ylim(0, 1)
    axo.set_ylabel("Sonnet-5 agreement (0–1)")
    axo.set_xlabel("depth of the RAW activation fed to the penultimate(L62)-trained reader")
    # two-part legend: colour = series, style = checkpoint
    leg1 = [Line2D([0], [0], color=SER[0][1], lw=2, label="workspace (J-lens)"),
            Line2D([0], [0], color=SER[1][1], lw=2, label="answer (continuation)")]
    leg2 = [Line2D([0], [0], color="0.35", ls=ls, marker=mk, label=title) for _, title, ls, mk in CK]
    l1 = axo.legend(handles=leg1, title="colour = metric", fontsize=8, loc="upper left")
    axo.add_artist(l1)
    axo.legend(handles=leg2, title="line style = checkpoint", fontsize=8, loc="lower right")

    fig.suptitle("Fed-layer sweep (n=64 disagreement set): future-lens variants read the WORKSPACE (J-lens) "
                 "far above the answer; RL-autoencoding reads it best\nQwen3.6-27B · Sonnet-5 judge · "
                 f"95% binomial CIs ({'shaded bands' if ci_style=='band' else 'error bars'})", fontsize=11)
    fig.subplots_adjust(top=0.88)
    out = f"{R}/fedlayer_combined_{ci_style}"
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=140, bbox_inches="tight")
    plt.close(fig); print("wrote", out + ".png")

for s in ("bars", "band"):
    build(s)
