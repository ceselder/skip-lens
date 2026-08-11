"""Workspace-agreement @ L42 — the skip-lens surfacing claim.

Feeding the middle layer (L42), does the lens readout track the model's WORKSPACE
(J-lens top-k concepts, `agree_jlens`) rather than the surface answer (`agree_answer`)?
And does it rise with futurelens pretraining data? Plus how the compositional cNLA lens
compares. Sonnet-judged agreement, n=321 items/checkpoint.

Reads   ~/shared/reports/skiplens-scaling/data/workspace_agreement.json
Writes  ~/shared/reports/skiplens-scaling/workspace_agreement.{png,pdf}
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/skiplens-scaling")
d = json.load(open(f"{RD}/data/workspace_agreement.json"))
bps = d.get("batch_per_step", 64)

fl = d["futurelens"]
its = sorted((int(k) for k in fl), key=int)
xs = [it * bps / 1000.0 for it in its]
aj = [fl[str(it)]["agree_jlens"] for it in its]
aa = [fl[str(it)]["agree_answer"] for it in its]

cnla = d.get("cnla", {})
c_aj = [v["agree_jlens"] for v in cnla.values() if v.get("agree_jlens") is not None]
c_aa = [v["agree_answer"] for v in cnla.values() if v.get("agree_answer") is not None]

fig, ax = plt.subplots(figsize=(8.6, 5.4))
ax.plot(xs, aj, "-o", color="#2060c0", lw=2.4, ms=6,
        label="futurelens — WORKSPACE agreement (agree_jlens)")
ax.plot(xs, aa, "--o", color="#d2691e", lw=2.0, ms=5,
        label="futurelens — surface-ANSWER agreement (agree_answer)")

if c_aj:
    cm, clo, chi = st.mean(c_aj), min(c_aj), max(c_aj)
    ax.axhspan(clo, chi, color="#2ca02c", alpha=0.12)
    ax.axhline(cm, color="#2ca02c", lw=2.0,
               label=f"cNLA — WORKSPACE agreement (mean {cm:.2f}, RL ckpts)")
    if c_aa:
        ax.axhline(st.mean(c_aa), color="#8fbf6f", ls=":", lw=1.6,
                   label=f"cNLA — answer agreement (mean {st.mean(c_aa):.2f})")

ax.set_ylim(0.0, 0.9)
ax.set_xlabel("Distinct futurelens pretraining examples seen (thousands)")
ax.set_ylabel("Agreement @ fed L42  (Sonnet judge, 0–1)")
ax.set_title("Fed L42, the lens surfaces the model's WORKSPACE ~2× the surface answer —\n"
             "rises with futurelens data, and the compositional cNLA lens surfaces it even more",
             fontsize=11, pad=12)
ax.legend(fontsize=8.5, loc="center right", framealpha=0.93)
ax.grid(alpha=0.25)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/workspace_agreement.{ext}", dpi=150, bbox_inches="tight")
print(f"wrote workspace_agreement.png/.pdf | FL agree_jlens {aj[0]:.3f}->{aj[-1]:.3f} "
      f"answer~{sum(aa)/len(aa):.3f} | cNLA jlens mean {sum(c_aj)/len(c_aj):.3f}")
