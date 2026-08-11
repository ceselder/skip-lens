"""AR read-position ablation: held-out reconstruction FVE vs training step.

Arms (token-matched: same 48k explanations, gold L62, lr/seed/1500-step budget):
  ANCHOR  — value head reads at the fixed '</text> <summary>' anchor (current design), single-idx loss.
  FINAL   — suffix stripped, reads at the explanation's last token, single-idx loss.
  ALLIDX  — no anchor; DENSE loss supervising reconstruction at every position (read at last token).
FVE = 1 - mse/baseline (higher = better). Eval read point is the last token for every arm.

Reads   ~/shared/reports/ar-readpos-ablation/data/ar_ablation.json
Writes  ~/shared/reports/ar-readpos-ablation/ar_ablation.{png,pdf}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RD = os.path.expanduser("~/shared/reports/ar-readpos-ablation")
d = json.load(open(f"{RD}/data/ar_ablation.json"))
STYLE = {
    "anchor": ("#3060c0", "-",  "ANCHOR — read at ⟨/text⟩⟨summary⟩ (Linear head)"),
    "final":  ("#d2691e", "--", "FINAL — read at explanation's last token"),
    "allidx": ("#2ca02c", "-.", "ALL-IDX — dense supervision every position"),
    "mlphead": ("#8e44ad", "-", "ANCHOR + deep residual-MLP head (700M, co-trained)"),
    "affine": ("#16a085", ":", "ANCHOR + just an affine (Linear+bias, identity-init)"),
    "mlp1": ("#e67e22", "-", "ANCHOR + 1-layer residual-MLP head"),
    "sumtok": ("#c0392b", "-", "ANCHOR + dedicated ⟨summary⟩ token (added to tokenizer)"),
    "freshblock": ("#111111", "-", "ANCHOR + fresh transformer block on top (identity-init)"),
}
order = [a for a in ("anchor", "final", "allidx", "mlphead")
         if a in d and d[a].get("heldout")]

fig, ax = plt.subplots(figsize=(8.4, 5.4))
finals = {}
for arm in order:
    ho = d[arm]["heldout"]                     # [(step, mse, fve%)]
    xs = [p[0] for p in ho]
    ys = [p[2] for p in ho]
    c, ls, lab = STYLE[arm]
    ax.plot(xs, ys, ls, color=c, marker="o", ms=5, lw=2.2, label=lab)
    ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), textcoords="offset points",
                xytext=(8, 0), fontsize=9, color=c, va="center")
    finals[arm] = ys[-1]

fa, ff = finals.get("anchor"), finals.get("final")
sub = ""
if fa is not None and ff is not None:
    sub = f"ANCHOR {fa:.1f}%  vs  FINAL {ff:.1f}%  (+{fa-ff:.1f})"
    if "allidx" in finals:
        sub += f"  vs  ALL-IDX {finals['allidx']:.1f}%"
ax.set_xlabel("AR-SFT training step (token-matched budget)")
ax.set_ylabel("Held-out reconstruction FVE (%)  —  higher = better activation reconstruction")
ax.set_title("Where the AR reads / how densely it's supervised → reconstruction fidelity\n" + sub,
             fontsize=11, pad=12)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.93)
ax.grid(alpha=0.25)
ax.margins(x=0.10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{RD}/ar_ablation.{ext}", dpi=150, bbox_inches="tight")
print("wrote ar_ablation.png/.pdf | arms:", {a: round(finals[a], 1) for a in finals})
