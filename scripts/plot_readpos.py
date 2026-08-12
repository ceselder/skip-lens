import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json
step=[250,500,750,1000,1250,1500]
last=[36.3,38.0,38.6,39.5,39.5,40.0]
summ=[36.4,37.6,38.5,39.2,39.8,39.8]
fig,ax=plt.subplots(figsize=(8.0,5.0))
ax.plot(step,last,"-o",color="#2a78d6",lw=2.4,ms=8,label="last-token read (40.0%)")
ax.plot(step,summ,"--s",color="#eb6834",lw=2.0,ms=7,label="dedicated <|summary|> anchor (39.8%)")
ax.set_xlabel("optimizer step (fixed 16-token spans, lr 1e-4)")
ax.set_ylabel("L62 activation reconstruction — FVE corrected (%)")
ax.set_title("At fixed 16-token spans the AR read position doesn't matter:\nlast-token vs dedicated <|summary|> anchor reconstruct identically (40.0% vs 39.8%)",
             fontsize=10,weight="bold")
ax.set_ylim(30,45); ax.grid(alpha=0.25); ax.legend(frameon=False,loc="lower right")
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("/home/celeste/skip-lens/scripts/readpos_span16.png",dpi=150,bbox_inches="tight")
plt.savefig("/home/celeste/skip-lens/scripts/readpos_span16.pdf",bbox_inches="tight")
json.dump({"step":step,"last_token_fve_corr":last,"summary_token_fve_corr":summ},open("/home/celeste/skip-lens/scripts/readpos_span16.json","w"),indent=2)
print("saved readpos_span16")
