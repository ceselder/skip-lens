"""Pastlens training-loss curve (next-token CE) over its 1-epoch run."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "/home/celeste/.claude/jobs/7b8dc830/tmp/pastlens_loss.csv"
rows = [l.strip().split(",") for l in open(CSV) if l.strip() and "," in l]
step = np.array([int(r[0]) for r in rows])
loss = np.array([float(r[1]) for r in rows])
w = 15
rm = np.convolve(loss, np.ones(w) / w, mode="valid")
x_rm = step[w - 1:]

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(step, loss, color="#e0c3b3", lw=0.9, alpha=0.8, label="raw (every 25 steps)")
ax.plot(x_rm, rm, color="#c0562f", lw=2.3, label=f"rolling mean ({w})")
ax.axhline(loss[0], ls=":", lw=1, color="#b0a89c")
ax.annotate(f"start {loss[0]:.1f}", (step[0], loss[0]), textcoords="offset points",
            xytext=(8, -4), fontsize=9, color="#8a8378")
ax.annotate(f"end ~{rm[-1]:.2f}", (x_rm[-1], rm[-1]), textcoords="offset points",
            xytext=(-8, 10), ha="right", fontsize=9, weight="bold", color="#3a3a3a")
ax.set_xlabel("optimizer step  (1 epoch = 12,124)")
ax.set_ylabel("training loss — next-token CE (nats)")
ax.set_title("Pastlens training loss: 5.9 → ~2.3, most of the drop in the first ~300 steps\n"
             "1 epoch, 776k on-policy pairs, eff batch 64, lr 1e-4 (rsLoRA r64/α16 all-modules)",
             fontsize=10.5, weight="bold")
ax.legend(frameon=False)
ax.grid(alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("/home/celeste/skip-lens/scripts/pastlens_loss.png", dpi=150, bbox_inches="tight")
plt.savefig("/home/celeste/skip-lens/scripts/pastlens_loss.pdf", bbox_inches="tight")
print(f"saved; start={loss[0]:.2f} min={loss.min():.2f} end_rollmean={rm[-1]:.2f}")
