import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

txt = open("/tmp/rlc/all.txt").read()
# split sections
def sect(name):
    m = re.search(rf"==={name}===(.*?)(?====|$)", txt, re.S)
    return m.group(1) if m else ""
STEP = re.compile(r"step (\d+) \| r (-?[\d.]+) \| FVE ([\d.]+)% \| ar_mse ([\d.]+) \| kl ([\d.]+) \| ent ([\d.]+) \| ext ([\d.]+)%")
def steps(s):
    return [dict(step=int(a), r=float(b), fve=float(c), kl=float(e), ent=float(f), ext=float(g))
            for a, b, c, d, e, f, g in STEP.findall(s)]
main = steps(sect("MAIN"))                      # 0..200 (produced iter_000200 = NLA-futurelens)
cont = [r for r in steps(sect("CONT")) if r["step"] > 200]   # 201..235 (bp64 continuation)
S = main + cont
ev = re.findall(r"\[eval@(\d+)\] reward (-?[\d.]+) \| FVE ([\d.]+)%", txt)          # held-out
tj = re.findall(r"\[text_judges@(\d+)\] unique_info ([\d.]+) coherence ([\d.]+) writing_quality ([\d.]+) specificity ([\d.]+) repetitiveness ([\d.]+)", txt)

x = [r["step"] for r in S]
def col(k): return [r[k] for r in S]
ev_x = [int(a) for a, _, _ in ev]; ev_fve = [float(c) for _, _, c in ev]; ev_r = [float(b) for _, b, _ in ev]
tj_x = [int(a) for a, *_ in tj]
coh = [float(t[2]) for t in tj]; wq = [float(t[3]) for t in tj]; rep = [float(t[5]) for t in tj]

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .3})
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
NLA = 200  # checkpoint taken for the NLA-futurelens

# 1) reward
ax[0,0].plot(x, col("r"), "-", color="#C44E52", lw=1.6, label="train reward (−recon MSE)")
if ev_x: ax[0,0].plot(ev_x, ev_r, "s", color="#8172B3", ms=6, label="held-out eval reward")
ax[0,0].set_title("RL reward"); ax[0,0].set_ylabel("reward"); ax[0,0].legend(fontsize=8)

# 2) reconstruction FVE
ax[0,1].plot(x, col("fve"), "-", color="#4C72B0", lw=1.6, label="train FVE (per-step)")
if ev_x: ax[0,1].plot(ev_x, ev_fve, "s", color="#8172B3", ms=6, label="held-out FVE")
ax[0,1].set_title("Reconstruction FVE (%)"); ax[0,1].set_ylabel("FVE %"); ax[0,1].legend(fontsize=8)

# 3) KL + entropy (twin)
a3 = ax[1,0]; a3.plot(x, col("kl"), "-", color="#DD8452", lw=1.6, label="KL(policy‖ref)")
a3.set_ylabel("KL", color="#DD8452"); a3.tick_params(axis="y", labelcolor="#DD8452")
a3b = a3.twinx(); a3b.plot(x, col("ent"), "-", color="#55A868", lw=1.4, label="entropy")
a3b.set_ylabel("entropy", color="#55A868"); a3b.tick_params(axis="y", labelcolor="#55A868"); a3b.grid(False)
a3.set_title("KL drift & policy entropy")

# 4) text-judge quality (continuation only)
a4 = ax[1,1]
if tj_x:
    a4.plot(tj_x, coh, "-o", color="#C44E52", label="coherence (1–10)")
    a4.plot(tj_x, wq, "-s", color="#4C72B0", label="writing quality")
    a4.plot(tj_x, rep, "-^", color="#8172B3", label="repetitiveness")
    a4.set_ylim(0, 10)
a4.set_title("Sonnet judge text quality (eval)"); a4.set_ylabel("score (1–10)"); a4.legend(fontsize=8)

for a in [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]:
    a.set_xlabel("RL step"); a.axvline(NLA, color="k", ls=":", lw=1, alpha=.6)
ax[0,0].annotate("iter_000200\n= NLA-futurelens", (NLA, ax[0,0].get_ylim()[0]),
                 fontsize=7, ha="center", va="bottom", color="k")

fig.suptitle("NLA-futurelens RL training (reconstruction GRPO on Qwen3.6-27B L62): "
             "FVE climbs 19%→~55% while readout coherence stays floored (~2/10)\n"
             "steps 0–200 = main run (→ iter_000200); 200–235 = bp64 continuation", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
R = "/home/celeste/multi-token-jlens-nla-lastlayer/results"
for e in ("png", "pdf"):
    fig.savefig(f"{R}/nla_rl_curves.{e}", dpi=140, bbox_inches="tight")
print("steps:", len(S), "| eval pts:", len(ev_x), "| judge pts:", len(tj_x))
print("FVE:", col('fve')[0], "->", col('fve')[-1], "| reward:", col('r')[0], "->", col('r')[-1])
print("wrote", f"{R}/nla_rl_curves.png")
