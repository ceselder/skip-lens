"""Does the futurelens AV training loss follow a power-law scaling curve?

Fit L(N) = E + A * N^(-alpha)  (irreducible floor E + power law in examples-seen N),
which is a straight line on log-log axes once E is subtracted. Grid over E (no scipy dep);
for each E, linear-regress log(L_binned - E) vs log(N) and keep the best R^2.

Reads   ~/shared/reports/skiplens-scaling/data/av_trainloss.json
Writes  ~/shared/reports/skiplens-scaling/av_scaling_fit.{png,pdf}
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-scaling")
d = json.load(open(f"{RD}/data/av_trainloss.json"))
bps = d["batch_per_step"]
steps = np.array(d["steps"], dtype=float)
loss = np.array(d["loss"], dtype=float)
N = steps * bps                                            # examples seen
m = N > 0
N, loss = N[m], loss[m]

# log-spaced bins of N -> mean loss (denoise before fitting)
edges = np.logspace(np.log10(N.min()), np.log10(N.max()), 40)
idx = np.digitize(N, edges)
bN, bL = [], []
for b in range(1, len(edges)):
    sel = idx == b
    if sel.sum() >= 3:
        bN.append(N[sel].mean()); bL.append(loss[sel].mean())
bN, bL = np.array(bN), np.array(bL)

# grid over irreducible floor E; fit log(L-E) = log A - alpha log N
best = None
for E in np.linspace(0.0, bL.min() - 0.02, 200):
    y = np.log(bL - E)
    x = np.log(bN)
    A_ = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A_, y, rcond=None)
    yhat = A_ @ [slope, intercept]
    ss_res = ((y - yhat) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    if best is None or r2 > best[0]:
        best = (r2, E, -slope, math.exp(intercept))
r2, E, alpha, A = best
Lfit = lambda n: E + A * n ** (-alpha)


def extrap(mult):
    n = N.max() * mult
    return n, Lfit(n)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2))

# panel 1: loss vs N, log-log, with fit + extrapolation
ax1.scatter(bN / 1e3, bL, s=16, color="#3060c0", alpha=0.8, label="binned training loss")
ng = np.logspace(np.log10(N.min()), np.log10(N.max() * 20), 200)
ax1.plot(ng / 1e3, Lfit(ng), "-", color="#c0392b", lw=2.2,
         label=f"fit  L = {E:.2f} + {A:.2f}·N^(−{alpha:.3f})")
ax1.axhline(E, color="#888", ls=":", lw=1.3, label=f"irreducible floor E = {E:.2f} nats")
for mult, lab in [(10, "10×"), (20, "20×")]:
    n, l = extrap(mult); ax1.annotate(f"{lab}: {l:.2f}", (n / 1e3, l), fontsize=8, color="#c0392b")
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("examples seen (thousands, log)"); ax1.set_ylabel("AV training loss (nats, log)")
ax1.set_title("Loss vs data (log–log) with power-law fit + extrapolation")
ax1.legend(fontsize=8, loc="upper right"); ax1.grid(alpha=0.25, which="both")

# panel 2: (L - E) vs N log-log -> should be a straight line if power-law
ax2.scatter(bN / 1e3, bL - E, s=16, color="#2ca02c", alpha=0.8, label="L − E (binned)")
ax2.plot(bN / 1e3, A * bN ** (-alpha), "-", color="#c0392b", lw=2.2,
         label=f"slope −{alpha:.3f}  (R² = {r2:.3f})")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("examples seen (thousands, log)"); ax2.set_ylabel("L − E  (nats above floor, log)")
ax2.set_title("Power-law test: L−E is straight in log–log")
ax2.legend(fontsize=8.5, loc="lower left"); ax2.grid(alpha=0.25, which="both")

verdict = ("clean power law" if r2 > 0.97 else "roughly power-law" if r2 > 0.9 else "NOT a clean power law")
fig.suptitle(f"AV training loss IS a power law in data ({verdict}): α={alpha:.3f}, floor E={E:.2f} nats, R²={r2:.3f}",
             fontsize=12, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/av_scaling_fit.{ext}", dpi=150, bbox_inches="tight")
print(f"fit: E={E:.3f} alpha={alpha:.3f} A={A:.3f} R2={r2:.4f} | "
      f"10x->{extrap(10)[1]:.3f} 20x->{extrap(20)[1]:.3f} nats")
