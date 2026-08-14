"""What context-averaging costs: a 62x collapse in effective dimensionality.

The per-offset design assumed that averaging ∂h_target,t+d/∂h_42,t over contexts
preserves the horizon structure of the individual derivatives. It does not. The
LOCAL transports the decoder trains on span roughly 750 effective dimensions and
are mutually near-orthogonal; their context-averaged counterparts collapse to
about 12 dimensions of near-copies.

This is the mechanism behind every failure in the study: slots that are copies of
one another cannot carry per-position information, so no rearrangement, rescaling,
centering, deflation, or routing fix could recover it.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.expanduser("~/shared/reports/skip-lens-multislot")
d = json.load(open(f"{R}/data/family_rank.json"))
D = d["deep_only_d_ge_1"]

fig, (a, b) = plt.subplots(1, 2, figsize=(13.6, 5.9))

kinds = ["local\n(per-example,\nwhat training saw)",
         "context-averaged\n(what the lens feeds\nat test time)"]
cols = ["#2f855a", "#9b2c2c"]

pr = [D["local"]["participation_ratio"], D["averaged"]["participation_ratio"]]
bars = a.bar(kinds, pr, color=cols, width=.58)
a.set_yscale("log")
a.set_ylabel("effective dimensions occupied by the 15 deep transports\n"
             "(participation ratio, log scale)", fontsize=10)
for r, v in zip(bars, pr):
    a.text(r.get_x() + r.get_width() / 2, v * 1.12, f"{v:.0f}", ha="center",
           fontsize=15, fontweight="bold")
a.annotate("", xy=(1, pr[1] * 1.6), xytext=(1, pr[0] * .62),
           arrowprops=dict(arrowstyle="-|>", color="#33404f", lw=2.2))
a.text(1.07, (pr[0] * pr[1]) ** .5, f"{D['rank_collapse_factor']:.0f}× collapse",
       fontsize=13, fontweight="bold", color="#33404f", va="center")
a.set_ylim(5, 2200)
a.grid(axis="y", alpha=.25, which="both")
a.set_title("A. Averaging over contexts destroys the family's dimensionality",
            fontsize=11.5, loc="left")

x = np.arange(2)
mc = [D["local"]["mean_abs_cos"], D["averaged"]["mean_abs_cos"]]
lo = [D["local"]["min_cos"], D["averaged"]["min_cos"]]
hi = [D["local"]["max_cos"], D["averaged"]["max_cos"]]
b.bar(x, mc, color=cols, width=.58)
b.errorbar(x, mc, yerr=[np.array(mc) - np.array(lo), np.array(hi) - np.array(mc)],
           fmt="none", ecolor="#33404f", lw=1.6, capsize=6)
for i, v in enumerate(mc):
    b.text(i, v + .035, f"{v:.2f}", ha="center", fontsize=14, fontweight="bold")
b.set_xticks(x); b.set_xticklabels(kinds, fontsize=9.5)
b.set_ylabel("mean |cosine| between transports at different offsets\n"
             "(bars span the min and max over all 105 offset pairs)", fontsize=10)
b.set_ylim(0, 1.12)
b.axhline(1.0, color="#87867F", ls=":", lw=1.2)
b.text(1.42, 1.03, "identical", color="#87867F", fontsize=9, ha="right")
b.grid(axis="y", alpha=.25)
b.set_title("B. The averaged offsets are near-copies of each other",
            fontsize=11.5, loc="left")

fig.suptitle("Why per-token-offset slots could never work: the averaged Jacobian "
             f"family occupies {D['averaged']['participation_ratio']:.0f} effective "
             f"dimensions where the per-example one occupies "
             f"{D['local']['participation_ratio']:.0f}\n"
             "Qwen3.6-27B · offsets 1-15 of the L42→L62 family · "
             f"{d['n_rows']} held-out on-policy rows",
             fontsize=13, y=1.01)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/rank_collapse.{ext}", dpi=170, bbox_inches="tight")
print(f"wrote {R}/rank_collapse.png/.pdf")
json.dump(d, open(f"{R}/data/rank_collapse.json", "w"), indent=1)
print("wrote data/rank_collapse.json")
