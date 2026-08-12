import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json

# eval_truncres_fair.py, held-out doc-disjoint val (n=1000), L62. For each span length k,
# wrap the first-k on-policy span tokens in the template and read at either:
#   ANCHOR  = last prompt token (the <summary> read position both were... last-token WAS trained on)
#   CONTENT = last CONTENT token (causal-identical to reading that token in the full prompt)
# Both ARs trained on FIXED 16-token spans. Three curves tell the whole story:
k = list(range(1, 17))
last_anchor  = [-26.1,-7.4,2.0,8.4,13.1,16.5,19.3,21.3,23.4,25.1,26.4,27.7,28.6,29.6,30.6,31.1]
last_content = [-57.4,-46.4,-42.4,-40.6,-39.6,-36.9,-36.9,-36.1,-37.4,-36.6,-35.9,-37.0,-36.7,-36.4,-35.9,-36.4]
dense_content= [-17.5,-2.3,4.7,10.1,13.6,16.2,18.5,20.4,22.4,23.9,25.0,26.2,27.0,27.9,28.7,29.4]
dense_anchor = [-24.0,-4.6,4.1,9.9,13.9,16.9,19.2,21.1,23.1,24.6,25.8,27.1,27.7,28.6,29.5,30.1]  # for json/reference

fig,ax=plt.subplots(figsize=(9.0,5.4))
ax.axhline(0,color="#b0a89c",lw=1)
ax.plot(k,dense_content,"-o",color="#1baf7a",lw=2.5,ms=6,
        label="dense --ar-all-idx, read @ content token")
ax.plot(k,last_anchor,"-s",color="#2f6fc0",lw=2.3,ms=6,
        label="last-token AR, read @ its anchor")
ax.plot(k,last_content,"--D",color="#c0392f",lw=2.3,ms=6,
        label="last-token AR, read @ content token")

ax.annotate("read at a position it was NEVER trained on ->\nOOD: worse-than-mean at every k, even k=16\n(length in-distribution, only read position moved)",
            (8,-36.1),textcoords="offset points",xytext=(20,34),fontsize=8.5,color="#c0392f",
            arrowprops=dict(arrowstyle="->",color="#c0392f",lw=1.2))
ax.annotate("read where it WAS trained -> fine;\ndense reads fine at BOTH positions",
            (13,28.6),textcoords="offset points",xytext=(-6,-78),fontsize=8.5,color="#333",
            arrowprops=dict(arrowstyle="->",color="#333",lw=1.1))

ax.set_xlabel("span tokens fed to the reader (k)   —   both ARs trained on fixed 16-token spans")
ax.set_ylabel("L62 reconstruction FVE, norm-constrained (%)")
ax.set_title("What --ar-all-idx actually buys is READ-POSITION robustness, not better reconstruction:\n"
             "a last-token AR is fine at its trained anchor but OOD (worse-than-mean) at the content token;\n"
             "the dense AR reads fine anywhere. Held-out doc-disjoint val, on-policy spans, L62.",
             fontsize=9.0,weight="bold")
ax.set_xlim(0.5,16.5); ax.set_ylim(-60,40); ax.set_xticks(range(2,17,2))
ax.grid(alpha=0.25); ax.legend(frameon=False,loc="lower right",fontsize=9)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("/home/celeste/skip-lens/scripts/truncres_span16.png",dpi=150,bbox_inches="tight")
plt.savefig("/home/celeste/skip-lens/scripts/truncres_span16.pdf",bbox_inches="tight")
json.dump({"span_tokens_k":k,"last_token_anchor":last_anchor,"last_token_content":last_content,
           "dense_all_idx_content":dense_content,"dense_all_idx_anchor":dense_anchor,
           "eval":"eval_truncres_fair.py held-out doc-disjoint val n=1000, L62",
           "note":"both ARs trained on FIXED 16-tok spans. last-token @ content = read-position OOD (flat ~-37% even at k=16). dense fine at content AND anchor. length generalizes at anchor; read-position does NOT for last-token."},
          open("/home/celeste/skip-lens/scripts/truncres_span16.json","w"),indent=2)
print("saved 3-curve read-position-OOD truncres_span16")
