import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json

# nla/train_rl_self_contained.py GRPO run rl_av12_frozenAR_kl03_peak (β_KL 0.03, 2048 rollouts/step,
# frozen truncation-resistant AR reward, ≤12-tok rollouts). Held-out doc-disjoint av12_val.
step   = list(range(13))
fve_tr = [17.2,16.2,19.9,26.7,33.4,38.3,34.0,33.7,35.1,34.9,39.3,41.8,38.3]
kl     = [0.00,0.25,0.23,0.22,0.50,0.39,0.42,0.52,0.55,0.57,0.55,0.70,0.77]
ev_s   = [0,5,10];  ev_fve = [6.6,29.6,33.1]   # held-out eval (the truth)

fig,ax=plt.subplots(figsize=(8.8,5.0)); ax2=ax.twinx()
ax.plot(step,fve_tr,"-o",color="#1baf7a",lw=2.4,ms=5,label="train reward-FVE")
ax.plot(ev_s,ev_fve,"-D",color="#2f6fc0",lw=2.6,ms=9,label="HELD-OUT eval FVE (the truth)")
ax2.plot(step,kl,"--",color="#c0562f",lw=1.8,label="KL to ref (right axis)")
ax.axhline(6.6,color="#b0a89c",ls=":",lw=1.2)
ax.text(11.5,7.6,"untrained AV baseline 6.6%",fontsize=8,color="#6b6357",ha="right")
ax.plot([10],[33.1],marker="*",ms=20,color="#2f6fc0",zorder=5)
ax.text(10,34.6,"best ckpt iter_000010\n33.1% held-out (5x baseline)",fontsize=8.2,color="#2f6fc0",ha="center")

ax.set_xlabel("RL step  (2048 rollouts each, ~21 min/step, HF generate — NOT vLLM)")
ax.set_ylabel("L62 reconstruction FVE, norm-constrained (%)")
ax2.set_ylabel("KL(policy || AV-SFT reference)")
ax.set_title("Futurelens-NLA RL vs frozen AR (β_KL 0.03): held-out reconstruction FVE\n"
             "6.6% -> 33.1% in 10 steps; KL climbs steadily (β 0.03 is a loose leash)",
             fontsize=9.5,weight="bold")
ax.set_ylim(0,45); ax2.set_ylim(0,1.0); ax.set_xlim(-0.3,12.3); ax.set_xticks(range(0,13,2))
ax.grid(alpha=0.25)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,frameon=False,loc="lower right",fontsize=8.6)
for s in ("top",): ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("/home/celeste/skip-lens/scripts/rl_trajectory.png",dpi=150,bbox_inches="tight")
plt.savefig("/home/celeste/skip-lens/scripts/rl_trajectory.pdf",bbox_inches="tight")
json.dump({"step":step,"train_fve":fve_tr,"kl":kl,"heldout_eval_step":ev_s,"heldout_eval_fve":ev_fve,
           "hypers":{"beta_kl":0.03,"batch_prompts":256,"group_size":8,"rollouts_per_step":2048,
                     "max_new_tokens":12,"policy_lr":1e-4,"quant":"none_bf16","generation":"HF generate (no vLLM)",
                     "sec_per_step":1265,"frozen_AR":"ar12_allidx_scaled_L62","policy_init":"av12_scaled_L62_lr3e5"}},
          open("/home/celeste/skip-lens/scripts/rl_trajectory.json","w"),indent=2)
print("saved rl_trajectory")
