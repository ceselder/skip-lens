"""Plot the fed-layer sweep: agreement with the J-lens [workspace proxy] and with the actual
continuation [the answer] vs the depth of the activation fed to the penultimate (L62) AO.
Faithful intermediate reader -> J-lens curve high. Optional --jac-scores adds a third curve:
agreement with the J-lens when J_{l->62}.h_l (Jacobian-mapped, i.e. the J-lens's own input) is
fed instead of the raw h_l."""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--jac-scores", default=None, help="optional scores json from the J.h_l (Jacobian) arm")
ap.add_argument("--out-stem", required=True)
ap.add_argument("--n-layers", type=int, default=64)
args = ap.parse_args()

d = json.load(open(args.scores))["by_fed_layer"]
layers = sorted(int(l) for l in d)                          # shallow -> deep, left to right
x = list(range(len(layers)))
xlabels = [f"L{l}\n({round(100*l/args.n_layers)}%)" for l in layers]

dj = None
if args.jac_scores and os.path.exists(args.jac_scores):
    dj = json.load(open(args.jac_scores))["by_fed_layer"]

def series(dd, key, lays):
    return [dd[str(l)].get(key) if str(l) in dd else None for l in lays]

CLAY, TEAL, PURP = "#c85a3c", "#3f8fb0", "#7a5aa0"
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(x, series(d, "agree_jlens", layers), yerr=[v or 0 for v in series(d, "agree_jlens_sem", layers)],
            fmt="-o", color=CLAY, lw=2.2, ms=7, capsize=3.5,
            label="raw h$_\\ell$ → J-lens agreement  (workspace proxy)")
ax.errorbar(x, series(d, "agree_answer", layers), yerr=[v or 0 for v in series(d, "agree_answer_sem", layers)],
            fmt="--s", color=TEAL, lw=2.0, ms=6, capsize=3.5,
            label="raw h$_\\ell$ → actual-continuation agreement  (the answer)")
for xi, v in enumerate(series(d, "agree_jlens", layers)):
    if v is not None:
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontweight="bold", color=CLAY, fontsize=6.5)

if dj is not None:
    jl = series(dj, "agree_jlens", layers)
    ax.errorbar(x, jl, yerr=[v or 0 for v in series(dj, "agree_jlens_sem", layers)],
                fmt="-^", color=PURP, lw=2.2, ms=7, capsize=3.5,
                label="J$_{\\ell\\to62}$·h$_\\ell$ → J-lens agreement  (Jacobian applied)")
    for xi, v in enumerate(jl):
        if v is not None:
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontweight="bold", color=PURP, fontsize=6.5)

ax.set_xticks(x); ax.set_xticklabels(xlabels, fontsize=8)
ax.set_xlabel("depth of the activation fed to the penultimate (L62) AO", fontsize=9)
ax.set_ylabel("Sonnet-5 agreement (0–1)", fontsize=9)
ax.tick_params(axis="y", labelsize=8)
ax.set_ylim(0, 1.0)
n = d[str(layers[0])].get("n")
ax.set_title(f"Fed-layer sweep (disagreement set, n={n}): does the penultimate AO track the workspace (J-lens) or the answer?",
             fontsize=9.5)
ax.legend(frameon=False, loc="lower center", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
os.makedirs(os.path.dirname(args.out_stem) or ".", exist_ok=True)
fig.savefig(args.out_stem + ".png", dpi=160); fig.savefig(args.out_stem + ".pdf")
out = {"fed_layers": layers, "agree_jlens_raw": series(d, "agree_jlens", layers),
       "agree_answer_raw": series(d, "agree_answer", layers), "n": n}
if dj is not None:
    out["agree_jlens_jac"] = series(dj, "agree_jlens", layers)
json.dump(out, open(args.out_stem + ".json", "w"), indent=2)
print(f"wrote {args.out_stem}.png/.pdf/.json")
