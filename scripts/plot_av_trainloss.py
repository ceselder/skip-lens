"""Futurelens AV training loss vs examples seen — does more data keep helping?

Raw per-step token-CE (nats) of the AV predicting the continuation from the injected L62
activation, over the 1-epoch 485k-example run. Rolling mean overlaid. The question: does the
loss keep dropping with more data, or plateau?

Reads   ~/shared/reports/skiplens-scaling/data/av_trainloss.json
Writes  ~/shared/reports/skiplens-scaling/av_trainloss.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-scaling")
d = json.load(open(f"{RD}/data/av_trainloss.json"))
bps = d["batch_per_step"]
steps = d["steps"]
loss = d["loss"]
xs = [s * bps / 1000.0 for s in steps]                      # examples seen (thousands)


def rolling(a, k=50):
    out = []
    for i in range(len(a)):
        lo = max(0, i - k)
        out.append(sum(a[lo:i + 1]) / (i - lo + 1))
    return out


sm = rolling(loss, 50)
fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot(xs, loss, color="#b0c4de", lw=0.6, alpha=0.6, label="raw per-step loss")
ax.plot(xs, sm, color="#c0392b", lw=2.4, label="rolling mean (50 steps)")
# plateau annotation: loss at ~32k examples (step 500) vs end
i500 = min(range(len(steps)), key=lambda i: abs(steps[i] - 500))
ax.axvline(xs[i500], color="#555", ls=":", lw=1)
ax.annotate(f"~{sm[i500]:.2f} nats by {xs[i500]:.0f}k examples\n(then flat → {sm[-1]:.2f} at {xs[-1]:.0f}k)",
            (xs[i500], sm[i500]), textcoords="offset points", xytext=(30, 40), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#555"))
ax.set_xlabel("Distinct pretraining examples seen (thousands)")
ax.set_ylabel("AV training loss — token CE of the continuation given L62 (nats)")
ax.set_title("The futurelens AV training loss plateaus at ~32k examples, then barely moves\n"
             "it is not data-limited — 15× more data buys ~0.13 nats; the single-activation bottleneck dominates",
             fontsize=10.8, pad=12)
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.25)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/av_trainloss.{ext}", dpi=150, bbox_inches="tight")
print(f"wrote av_trainloss.png/.pdf | start~{sm[0]:.2f} @500steps~{sm[i500]:.2f} end~{sm[-1]:.2f}")
