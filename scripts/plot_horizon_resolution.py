"""The headline: a context-averaged Jacobian is not horizon-resolved.

J̄⁽ᵈ⁾ is fit as the average of ∂h_target,t+d/∂h42,t, so slot d is supposed to be
about the token d ahead — the assumption the entire per-offset multi-slot design
rests on. These heatmaps test it directly in readout space: cell (d,e) is how
often the token actually at offset e appears in the top-50 of J̄⁽ᵈ⁾h, minus the
same measurement against a DIFFERENT row's continuation so that generic token
frequency is subtracted.

If the family were horizon-resolved the bright cells would run down the diagonal.
They run down column 0 instead, for both target layers. Every source offset is
mostly about the immediate next token, plus a flat diffuse future tail, with the
immediate part decaying as d grows.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = os.path.expanduser("~/shared/reports/skip-lens-multislot")
FAM = [("L42 → L62 (penultimate)", "horizon_matrix.json"),
       ("L42 → L63 (last layer, arm E's family)", "horizon_matrix_last.json")]

fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.1),
                         gridspec_kw={"width_ratios": [1, 1, 0.92]})
stats, mats = {}, {}
for ax, (title, fn) in zip(axes[:2], FAM):
    d = json.load(open(f"{R}/data/{fn}"))
    net = np.array(d["net"])
    mats[title] = net
    K = net.shape[0]
    im = ax.imshow(net, cmap="magma", vmin=0, vmax=0.21, aspect="auto")
    ax.plot([0, K - 1], [0, K - 1], color="#6ee7b7", lw=1.6, ls="--",
            label="where a horizon-resolved family would be bright")
    ax.set_xlabel("target offset e — the token actually at this position", fontsize=10)
    ax.set_ylabel("source offset d — which J̄⁽ᵈ⁾ was applied", fontsize=10)
    ax.set_title(title, fontsize=11.5)
    ax.set_xticks(range(0, K, 2)); ax.set_yticks(range(0, K, 2))
    ax.legend(loc="upper right", fontsize=8.5, framealpha=.85)
    fig.colorbar(im, ax=ax, fraction=.046, label="future-token recall above the\n"
                                                 "token-frequency floor")
    # honesty check: the on-diagonal mean is barely above off-diagonal, and that
    # edge is carried by d=0 alone. Report both with and without it.
    dg = [net[i, i] for i in range(K)]
    off = [net[i, j] for i in range(K) for j in range(K) if i != j]
    dg1 = [net[i, i] for i in range(1, K)]
    off1 = [net[i, j] for i in range(1, K) for j in range(1, K) if i != j]
    stats[title] = {
        "on_diagonal": float(np.mean(dg)), "off_diagonal": float(np.mean(off)),
        "on_diagonal_excl_d0": float(np.mean(dg1)),
        "off_diagonal_excl_d0": float(np.mean(off1)),
        "sources_peaking_own_horizon": int(d["n_sources_peaking_on_own_horizon"]),
        "sources_peaking_offset0": int(d["n_sources_peaking_at_offset_0"]),
        "col0_by_source": [float(net[i, 0]) for i in range(K)],
    }

ax = axes[2]
for (title, _), c in zip(FAM, ("#c05621", "#2b6cb0")):
    net = mats[title]
    K = net.shape[0]
    ax.plot(range(K), [net[d, 0] for d in range(K)], "o-", color=c, lw=2, ms=4.5,
            label=f"{title.split(' (')[0]}: immediate token (e=0)")
    ax.plot(range(K), [net[d, d] for d in range(K)], "s--", color=c, lw=1.5, ms=4,
            alpha=.65, label=f"{title.split(' (')[0]}: its OWN horizon (e=d)")
ax.set_xlabel("source offset d", fontsize=10)
ax.set_ylabel("future-token recall above the frequency floor", fontsize=10)
ax.set_title("Each J̄⁽ᵈ⁾ is about the NEXT token, not the token d ahead\n"
             "and that immediate content decays as d grows", fontsize=11.5)
ax.grid(alpha=.25)
ax.legend(fontsize=8, loc="upper right")
ax.axhline(0, color="#33404f", lw=.8)

fig.suptitle("Context-averaging destroys the horizon structure it is supposed to "
             "resolve: 15 of 16 per-offset Jacobians peak on the IMMEDIATE token\n"
             "Qwen3.6-27B · 384 held-out on-policy rows · top-50 readout through the "
             "unembedding · corrected against a shuffled context",
             fontsize=13, y=1.02)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{R}/horizon_resolution.{ext}", dpi=170, bbox_inches="tight")
print(f"wrote {R}/horizon_resolution.png/.pdf")

for t, v in stats.items():
    print(f"\n{t}")
    print(f"  on-diag {v['on_diagonal']:+.4f} vs off-diag {v['off_diagonal']:+.4f}")
    print(f"  excluding d=0: on-diag {v['on_diagonal_excl_d0']:+.4f} vs "
          f"off-diag {v['off_diagonal_excl_d0']:+.4f}  <- the edge was d=0 alone"
          if v["on_diagonal_excl_d0"] <= v["off_diagonal_excl_d0"] else
          f"  excluding d=0: on-diag {v['on_diagonal_excl_d0']:+.4f} vs "
          f"off-diag {v['off_diagonal_excl_d0']:+.4f}")
    print(f"  peaking on own horizon: {v['sources_peaking_own_horizon']}/16; "
          f"at offset 0: {v['sources_peaking_offset0']}/16")
json.dump(stats, open(f"{R}/data/horizon_resolution.json", "w"), indent=1)
print("\nwrote data/horizon_resolution.json")
