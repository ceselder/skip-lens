import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json

# Eval logs FVE indexed by d = distance-from-last-token (the read anchor). The template
# suffix "</text> <summary>" is m=6 tokens, so the anchor sits 6 tokens PAST the span end.
# => span tokens actually seen by the reader = clamp(16 + m - d, 0, 16) = clamp(22 - d, 0, 16).
#   d=0..6  -> full 16-token span seen (reads walk the 6-token suffix): the flat ~30% cap.
#   d=7..18 -> real truncation, 15 .. 4 span tokens.
# max_dist=18 + a 6-token suffix means we NEVER measured below 4 span tokens. All held-out val.
M = 6
last_d = [31.2,25.2,-10.2,-14.2,-3.1,-17.2,-38.1,-37.0,-36.2,-36.1,-36.7,-36.2,-37.2,-37.6,-36.4,-37.2,-37.3,-39.4,-41.0]
allx_d = [30.2,30.2,30.2,30.1,29.8,29.8,29.4,28.7,27.9,27.0,26.3,25.1,23.7,22.2,20.4,18.6,16.3,13.5, 9.9]

# dense curve over span tokens actually seen (d>=M => reads land inside the span)
seen, dense = [], []
for d in range(len(allx_d)):
    s = 16 + M - d
    if 4 <= s <= 16:
        seen.append(s); dense.append(allx_d[d])
order = sorted(range(len(seen)), key=lambda i: seen[i])
seen  = [seen[i]  for i in order]; dense = [dense[i] for i in order]

# suffix-plateau reads (d=0..M): full span seen, anchor walking the 6-token suffix
plat = [allx_d[d] for d in range(0, M+1)]           # ~30.2 .. 29.4

fig,ax=plt.subplots(figsize=(8.8,5.2))
ax.axhline(0,color="#b0a89c",lw=1)
ax.plot(seen,dense,"-o",color="#1baf7a",lw=2.6,ms=6,label="dense --ar-all-idx (reads at every prefix; a real truncation test)")
# suffix plateau (full span, reads through the 6-tok suffix)
ax.plot([16]*len(plat),plat,marker="o",ms=4,color="#1baf7a",alpha=.35,ls="none")
# last-token AR has exactly ONE valid read position: its trained anchor (= full span seen)
ax.plot([16],[31.2],marker="*",ms=18,color="#c0562f",ls="none",
        label="last-token AR (ONLY valid readout: its trained anchor = full span)")

ax.axvspan(2.2,4,color="#d9d2c6",alpha=.35)
ax.text(3.1,18,"below 4 tokens\nNOT measured\n(6-tok suffix +\nmax_dist=18)",fontsize=8,color="#6b6357",ha="center",va="center")
ax.annotate("last-token AR = dense AR at full span (~31% vs ~30%);\n"
            "it has NO defined readout at fewer tokens without\n"
            "re-encoding a shortened prompt (fair truncation curve = TODO)",
            (16,31.2),textcoords="offset points",xytext=(-14,-70),fontsize=8,color="#c0562f",ha="right",
            arrowprops=dict(arrowstyle="->",color="#c0562f",lw=1.1))

ax.set_xlabel("span tokens actually seen by the reader  (corrected for the 6-token template suffix)")
ax.set_ylabel("L62 reconstruction FVE, norm-constrained (%)")
ax.set_title("Dense reader is truncation-resistant: FVE rises ~concavely 4 tok (10%) -> 16 tok (30%), reads at ANY prefix.\n"
             "Last-token AR matches it at full span but has only one valid read position. Held-out doc-disjoint val, span16 on-policy.",
             fontsize=8.8,weight="bold")
ax.set_ylim(0,40); ax.set_xlim(2.2,16.8); ax.set_xticks(range(4,17,2))
ax.grid(alpha=0.25); ax.legend(frameon=False,loc="lower right",fontsize=8.5)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("/home/celeste/skip-lens/scripts/truncres_span16.png",dpi=150,bbox_inches="tight")
plt.savefig("/home/celeste/skip-lens/scripts/truncres_span16.pdf",bbox_inches="tight")
json.dump({"span_tokens_seen":seen,"dense_all_idx_fve":dense,
           "last_token_at_anchor_fve":31.2,"suffix_plateau_fve":plat,"suffix_tokens_m":M,
           "note":"held-out doc-disjoint val; x=span tokens seen=clamp(22-d,0,16); <4 tokens unmeasured (6-tok suffix + max_dist=18); last-token AR only has a valid readout at its trained anchor (full span) -- reads at other positions are untrained and were dropped as meaningless"},
          open("/home/celeste/skip-lens/scripts/truncres_span16.json","w"),indent=2)
print("saved corrected truncres_span16 (span-tokens-seen axis, m=6)")
