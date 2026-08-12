import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
J="/home/celeste/.claude/jobs/7b8dc830/tmp"
rw=[l.strip().split(",") for l in open(J+"/cnla_lh_reward.csv") if "," in l]
co=[l.strip().split(",") for l in open(J+"/cnla_lh_coh.csv") if "," in l]
rs=[int(a) for a,b in rw]; rv=[float(b)*100 for a,b in rw]
cs=[int(a) for a,b in co]; cv=[float(b) for a,b in co]
fig,ax=plt.subplots(figsize=(8.6,5.0))
ax.plot(rs,rv,"-",color="#2a78d6",lw=1.4,alpha=0.55)
ax.plot(rs,rv,"o",color="#2a78d6",ms=3,label="reconstruction reward / FVE (%)")
ax.set_xlabel("cNLA RL step (long-horizon run, 600 steps)")
ax.set_ylabel("reconstruction reward — FVE (%)",color="#2a78d6")
ax.set_ylim(0,60); ax.tick_params(axis="y",labelcolor="#2a78d6")
ax2=ax.twinx()
ax2.plot(cs,cv,"-s",color="#eb6834",lw=2.0,ms=7,label="coherence (Sonnet judge, 0–10)")
ax2.set_ylabel("coherence (0–10)",color="#eb6834"); ax2.set_ylim(0,10); ax2.tick_params(axis="y",labelcolor="#eb6834")
ax.set_title("cNLA long-horizon RL (600 steps): reward/FVE (~40–48%) and coherence (~4.5–5) both plateau\n(the full run the Aug-7 report never showed — it stopped at ~step 100)",fontsize=10,weight="bold")
for s in ("top",): ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
ax.legend(l1+l2,la1+la2,frameon=False,loc="lower center",fontsize=9)
plt.tight_layout()
plt.savefig("/home/celeste/skip-lens/scripts/cnla_longhorizon_600.png",dpi=150,bbox_inches="tight")
plt.savefig("/home/celeste/skip-lens/scripts/cnla_longhorizon_600.pdf",bbox_inches="tight")
print("saved cnla_longhorizon_600")
