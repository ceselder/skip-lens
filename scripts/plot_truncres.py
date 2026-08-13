import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json

# eval_truncres_fair.py, held-out doc-disjoint val (n=1000), L62. FVE vs span length k,
# each model read at its TRAINED position. Three regimes:
#   1) last-content AR  (no anchor, loss @ last content token only) read @ content  -> OOD below train len
#   2) dense --ar-all-idx (no anchor, loss @ every prefix)          read @ content  -> truncation-resistant
#   3) anchored last-token AR (</text> <summary> anchor)            read @ anchor   -> length-generalizes (ref)
# All trained on FIXED 16-token on-policy spans. k=16 = the trained length.
k = list(range(1, 17))
last_content = [-48.9,-32.0,-21.3,-13.3,-8.9,-4.6,-2.1,0.1,2.4,5.5,10.1,13.7,20.3,26.4,28.7,29.7]
dense_noanc  = [-17.2,-2.1,4.8,10.2,13.8,16.3,18.6,20.4,22.5,23.9,25.0,26.1,26.9,27.8,28.6,29.3]
anchored_lt  = [-26.1,-7.4,2.0,8.4,13.1,16.5,19.3,21.3,23.4,25.1,26.4,27.7,28.6,29.6,30.6,31.1]

fig,ax=plt.subplots(figsize=(9.1,5.4))
ax.axhline(0,color="#b0a89c",lw=1)
ax.plot(k,last_content,"--D",color="#c0392f",lw=2.4,ms=6,label="last-token AR, NO anchor  (read @ content token)")
ax.plot(k,dense_noanc,"-o",color="#1baf7a",lw=2.5,ms=6,label="dense --ar-all-idx, NO anchor  (read @ content token)")
ax.plot(k,anchored_lt,"-s",color="#2f6fc0",lw=2.2,ms=5,label="last-token AR, WITH template anchor  (read @ anchor)")

ax.set_xlabel("span tokens fed to the reader (k)   —   all trained on fixed 16-token on-policy spans")
ax.set_ylabel("L62 reconstruction FVE, norm-constrained (%)")
ax.set_title("Truncation-resistance confirmed: a last-token AR trained at ONE read position (bare content token)\n"
             "is OOD below its 16-token training length; the dense --ar-all-idx objective (causal-attn trick) OR a\n"
             "fixed template anchor both restore graceful reconstruction at every k. Held-out val, L62.",
             fontsize=8.8,weight="bold")
ax.set_xlim(0.5,16.5); ax.set_ylim(-52,36); ax.set_xticks(range(2,17,2))
ax.grid(alpha=0.25); ax.legend(frameon=False,loc="lower right",fontsize=8.6)
for s in ("top","right"): ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig("/home/celeste/skip-lens/scripts/truncres_span16.png",dpi=150,bbox_inches="tight")
plt.savefig("/home/celeste/skip-lens/scripts/truncres_span16.pdf",bbox_inches="tight")
json.dump({"span_tokens_k":k,"last_content_noanchor_read_content":last_content,
           "dense_allidx_noanchor_read_content":dense_noanc,"anchored_lasttoken_read_anchor":anchored_lt,
           "eval":"eval_truncres_fair.py held-out doc-disjoint val n=1000 L62; all trained on fixed 16-tok on-policy spans",
           "note":"last-token @ bare content token = OOD below train len (crater); dense all-idx OR anchor cue = truncation-resistant; all meet ~29-31% at k=16"},
          open("/home/celeste/skip-lens/scripts/truncres_span16.json","w"),indent=2)
print("saved 3-curve confirmed-truncation-resistance plot")
