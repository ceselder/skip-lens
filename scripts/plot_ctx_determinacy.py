"""Context-prefix determinacy sweep: how much of a layer-62 activation can be reconstructed
from the future span alone vs. span + N tokens of the context the model just read."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# corrected FVE (standard regression def) from the ctx-prefix AR sweep, all arms 1 epoch
ctx_tokens = [0, 32, 128, 256]
fve = [34.7, 70.7, 66.2, 63.6]
cos = [0.714, 0.884, 0.864, 0.853]
labels = ["span only", "+32", "+128", "+256 (full)"]
xs = list(range(len(ctx_tokens)))

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.axhline(fve[0], ls="--", lw=1.2, color="#b0a89c", zorder=1)
ax.plot(xs, fve, "-o", lw=2.6, ms=10, color="#c0562f", zorder=3)
for x, y, l in zip(xs, fve, labels):
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=10, weight="bold", color="#3a3a3a")
ax.annotate("span alone: 35%", (0, fve[0]), textcoords="offset points", xytext=(6, -18),
            ha="left", fontsize=9, color="#8a8378")
ax.set_xticks(xs)
ax.set_xticklabels([f"{c}\n{l}" if c == 0 else str(c) for c, l in zip(ctx_tokens, labels)], fontsize=10)
ax.set_xlabel("tokens of the model's read context given to the reconstructor (0 = future span only)", fontsize=10)
ax.set_ylabel("layer-62 activation reconstruction — FVE (%)", fontsize=10)
ax.set_title("A layer-62 activation is much more than its future span:\n"
             "the span alone reconstructs 35%; adding the context it just read nearly doubles it (71% at +32 tok)",
             fontsize=10.5, weight="bold")
ax.set_ylim(0, 82)
ax.grid(axis="y", alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("/home/celeste/skip-lens/scripts/ctx_determinacy.png", dpi=150, bbox_inches="tight")
plt.savefig("/home/celeste/skip-lens/scripts/ctx_determinacy.pdf", bbox_inches="tight")
json.dump({"ctx_tokens": ctx_tokens, "fve_corrected_pct": fve, "cos": cos,
           "note": "AR reconstruct L62 from span + N read-context tokens; 1 epoch each"},
          open("/home/celeste/skip-lens/scripts/ctx_determinacy.json", "w"), indent=2)
print("saved ctx_determinacy.png / .pdf / .json")
