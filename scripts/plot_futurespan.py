"""FVE vs future-span length: does the AR reconstruct a layer-62 activation better with more
future tokens? Each arm = AR trained on a FIXED future-span length, span-only, 1500 steps."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = [4, 8, 12, 16]
fve_corr = [25.7, 33.4, 37.6, 40.1]
fve_norm = [10.8, 22.0, 27.8, 31.2]
cos = [0.666, 0.708, 0.730, 0.742]

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(L, fve_corr, "-o", lw=2.6, ms=10, color="#2f7ec0", label="FVE (corrected, optimal-magnitude)")
ax.plot(L, fve_norm, "--s", lw=1.7, ms=7, color="#a9c2da", label="FVE (norm-constrained, as-logged)")
for x, y in zip(L, fve_corr):
    ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=10, weight="bold", color="#243b4a")
ax.annotate("still rising at 16 →\nlonger spans should help further", (16, 40.1),
            textcoords="offset points", xytext=(-12, -40), ha="right", fontsize=9, color="#c0562f")
ax.set_xticks(L)
ax.set_xlabel("future-span length given to the reconstructor (tokens)")
ax.set_ylabel("layer-62 activation reconstruction — FVE (%)")
ax.set_title("The activation reconstructor is span-length-limited:\n"
             "more future tokens → better reconstruction (26%→40% for 4→16 tok), still climbing",
             fontsize=10.5, weight="bold")
ax.set_ylim(0, 50)
ax.grid(alpha=0.25)
ax.legend(frameon=False, loc="lower right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("/home/celeste/skip-lens/scripts/futurespan_sweep.png", dpi=150, bbox_inches="tight")
plt.savefig("/home/celeste/skip-lens/scripts/futurespan_sweep.pdf", bbox_inches="tight")
json.dump({"future_span_tokens": L, "fve_corrected_pct": fve_corr, "fve_norm_pct": fve_norm, "cos": cos},
          open("/home/celeste/skip-lens/scripts/futurespan_sweep.json", "w"), indent=2)
print("saved futurespan_sweep.{png,pdf,json}")
