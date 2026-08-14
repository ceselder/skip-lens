"""E[J] is the wrong estimator — and fixing it splits vector space from readout space.

The J-lens readout is J̄ = E[J], the plain average of per-example Jacobians. The
best FIXED linear map from h to Jh is not that average but the regression solution
W* = E[(Jh)hᵀ]·E[hhᵀ]⁻¹, and the two agree only if J is uncorrelated with h — which
it cannot be, since h encodes the context whose Jacobian J is.

Panel A: W* nearly doubles alignment to the transport a decoder trains on, and it
is not mean-fitting — the gain survives (indeed grows under) mean removal, and the
output is half as mean-dominated.

Panel B: the ridge penalty trades alignment against effective rank, with a clear
interior optimum.

Panel C: the catch. W* wins vector space and LOSES readout space, because ridge
minimises total squared error while a logit-lens readout sees only the small
unembedding-readable component of the transport.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.expanduser("~/shared/reports/skip-lens-multislot")
fit = json.load(open(f"{R}/data/fit_report.json"))
hon = json.load(open(f"{R}/data/honest_eval.json"))
rd_j = json.load(open(f"{R}/data/pool_readout.json"))
rd_w = json.load(open(f"{R}/data/pool_readout_wreg.json"))

BASE, REG = "plain averaged Jacobian E[J]", "ridge regression W*"
fig, (a, b, c) = plt.subplots(1, 3, figsize=(17.4, 6.0))

lbl = ["cos to the local\ntransport (raw)", "cos to it\nCENTERED",
       "cos of output to\nits own mean", "effective rank,\ncentered (÷100)"]
bv = [hon[BASE]["cos_pooled_raw"], hon[BASE]["cos_pooled_centered"],
      hon[BASE]["cos_pooled_to_own_mean"], hon[BASE]["eff_rank_deep_centered"] / 100]
wv = [hon[REG]["cos_pooled_raw"], hon[REG]["cos_pooled_centered"],
      hon[REG]["cos_pooled_to_own_mean"], hon[REG]["eff_rank_deep_centered"] / 100]
x = np.arange(len(lbl))
a.bar(x - .19, bv, .38, label="plain E[J] (the J-lens estimator)", color="#7c8ba1")
a.bar(x + .19, wv, .38, label="ridge W* (the optimal fixed map)", color="#c05621")
for i, (u, v) in enumerate(zip(bv, wv)):
    a.text(i - .19, u + .012, f"{u:.2f}", ha="center", fontsize=10)
    a.text(i + .19, v + .012, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
a.set_xticks(x); a.set_xticklabels(lbl, fontsize=9)
a.legend(fontsize=9, loc="upper left")
a.grid(axis="y", alpha=.25)
a.set_ylim(0, .78)
a.set_title("A. The gain is real: it survives mean removal, and\nW* is half as "
            "mean-dominated as E[J]", fontsize=11, loc="left")

lams = sorted(fit["by_lambda"].keys(), key=float)
cp = [fit["by_lambda"][k]["cos_pooled"] for k in lams]
er = [fit["by_lambda"][k]["eff_rank_deep"] for k in lams]
b.plot(range(len(lams)), cp, "o-", color="#c05621", lw=2.2, ms=7,
       label="cos to the pooled transport")
b.axhline(fit["baseline"]["cos_pooled"], color="#7c8ba1", ls="--", lw=1.8,
          label=f"plain E[J] ({fit['baseline']['cos_pooled']:.3f})")
b.set_xticks(range(len(lams)))
b.set_xticklabels([f"{float(k):g}" for k in lams], fontsize=9)
b.set_xlabel("ridge penalty λ", fontsize=10)
b.set_ylabel("cosine to the pooled local transport", fontsize=10, color="#c05621")
b.set_ylim(0, .40)
b.grid(alpha=.25)
b2 = b.twinx()
b2.plot(range(len(lams)), er, "s:", color="#2b6cb0", lw=1.8, ms=6,
        label="effective rank")
b2.axhline(fit["baseline"]["eff_rank_deep"], color="#2b6cb0", ls=":", lw=1.1, alpha=.5)
b2.set_ylabel("effective rank of the deep transports", fontsize=10, color="#2b6cb0")
b2.set_ylim(0, 60)
h1, l1 = b.get_legend_handles_labels()
h2, l2 = b2.get_legend_handles_labels()
b.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="lower center")
b.set_title(f"B. Alignment and rank trade off, optimum at λ={fit['best_lambda']:g}\n"
            f"{fit['n_fit_rows']:,} fit rows, held-out shard", fontsize=11, loc="left")


def best_net(rows):
    return max(r["span_net"] for r in rows)


def named(rows, nm):
    return next(r["span_net"] for r in rows if r["weights"] == nm)


pairs = [("vector space\ncos to local transport",
          hon[BASE]["cos_pooled_centered"], hon[REG]["cos_pooled_centered"], 1.0),
         ("readout space\nspan recall (×5)",
          best_net(rd_j) * 5, best_net(rd_w) * 5, 5.0)]
x = np.arange(2)
c.bar(x - .19, [p[1] for p in pairs], .38, color="#7c8ba1", label="plain E[J]")
c.bar(x + .19, [p[2] for p in pairs], .38, color="#c05621", label="ridge W*")
for i, p in enumerate(pairs):
    c.text(i - .19, p[1] + .008, f"{p[1] / p[3]:.3f}", ha="center", fontsize=10)
    c.text(i + .19, p[2] + .008, f"{p[2] / p[3]:.3f}", ha="center", fontsize=10,
           fontweight="bold")
    win = "W* wins 2×" if p[2] > p[1] else "E[J] wins 1.5×"
    c.text(i, max(p[1], p[2]) + .045, win, ha="center", fontsize=10.5,
           fontweight="bold", color="#9b2c2c" if p[1] > p[2] else "#2f855a")
c.set_xticks(x); c.set_xticklabels([p[0] for p in pairs], fontsize=9.5)
c.legend(fontsize=9, loc="upper right")
c.grid(axis="y", alpha=.25)
c.set_ylim(0, .46)
c.set_title("C. The catch: opposite verdicts by readout type.\nRidge optimises L2, "
            "the logit lens reads only direction", fontsize=11, loc="left")

fig.suptitle("The averaged Jacobian is not the best fixed operator: regression "
             "nearly doubles alignment to the transport a decoder trains on\n"
             "but loses to it under a logit-lens readout — Qwen3.6-27B, "
             f"L42→L62, {fit['n_fit_rows']:,} fit rows, held-out evaluation",
             fontsize=12.5, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/regression_lens.{ext}", dpi=170, bbox_inches="tight")
print(f"wrote {R}/regression_lens.png/.pdf")

summary = {"honest_eval": hon, "fit_report": fit,
           "readout_best": {"plain_EJ": best_net(rd_j), "ridge_W": best_net(rd_w)},
           "readout_by_config": {r["weights"]: {"EJ": named(rd_j, r["weights"]),
                                                "W": named(rd_w, r["weights"])}
                                 for r in rd_j if any(q["weights"] == r["weights"]
                                                      for q in rd_w)}}
json.dump(summary, open(f"{R}/data/regression_lens.json", "w"), indent=1)
print("wrote data/regression_lens.json")
