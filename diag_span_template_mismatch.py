"""Does a SPAN-averaged Jacobian sit closer to the global one than a local one does?

The design under test: instead of training on a single context's Jacobian (arm I),
train on the Jacobian averaged over many contexts in which the SAME span is a
natural continuation. Both train and test vectors are then "an averaged Jacobian
applied to a real activation", and the only train/test difference is which
contexts the average ran over — S-specific vs all — rather than one context vs all.

That predicts a smaller mismatch, and the prediction is checkable before training
anything. For a held-out real context with activation h whose continuation is S:

    cos( Jbar h , Jbar_S h )      the mismatch under this design
    cos( Jbar h , v_local(h) )    arm I's mismatch, measured at 0.353

Jbar_S h is computed without ever forming a matrix: it is the mean over generated
contexts j of J_local(ctx_j) h, and each term is one double-VJP with tangent h
seeded at ctx_j's last position. One pass yields all 16 offsets, so the cost is M
passes per test row. The DOUBLE-VJP (reverse-over-reverse) backend is required, not
merely convenient: forward-AD disagrees with reverse-AD through this model's hybrid
DeltaNet stack, and the Jbar being compared against was fitted with reverse-mode
comb cotangents, so the two must share reverse semantics. Forward-mode also fails
outright here, since the fla gated-delta-rule kernel is a custom autograd.Function
without setup_context.

Offsets are pooled over d < span_len, because offset d is the state that predicts
span token d — pooling further would describe tokens outside the span, which is the
alignment bug found in the original arm K.

Also reports P(S | generated context) so contexts where the model would not
actually say S can be filtered, and the same mismatch restricted to survivors.
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/skip-lens")
from pretrain.collect_jvp_transport import N_OFFSETS, dvjp_transports

ap = argparse.ArgumentParser()
ap.add_argument("--templates", default="/workspace/data/span_templates/prototype.jsonl")
ap.add_argument("--shards", default="/workspace/data/spans_jvp/shard_5_jvp.parquet")
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--span-len", type=int, default=4)
ap.add_argument("--max-spans", type=int, default=150)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--min-logprob", type=float, default=-12.0,
                help="drop generated contexts whose total logprob of S is below this")
ap.add_argument("--out", default="/workspace/results/span_template_mismatch.json")
args = ap.parse_args()
dev = "cuda"
L = args.span_len
cos = torch.nn.functional.cosine_similarity

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
model = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

JP = None
for d in range(L):
    M = torch.from_numpy(np.load(
        f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy")).float().to(dev)
    JP = M if JP is None else JP + M
print(f"global Jbar pooled over d<{L}: |JP|={float(JP.norm()):.2f}", flush=True)

tpl = [json.loads(x) for x in open(args.templates)][: args.max_spans]
by_span = {tuple(t["span_ids"]): t for t in tpl}
print(f"{len(tpl)} spans with templates", flush=True)

# real held-out rows: give us h42 and the stored local transport for the SAME span
t = pq.read_table(args.shards, columns=["rollout_token_ids", "activation_vector",
                                        "transported_vectors", "h42_recompute_cosine"])
real = {}
for r in t.slice(0, 60000).to_pylist():
    if r["h42_recompute_cosine"] < 0.99 or not r["rollout_token_ids"]:
        continue
    k = tuple(int(x) for x in r["rollout_token_ids"][:L])
    if k in by_span and k not in real:
        real[k] = r
print(f"{len(real)} spans matched to a real held-out row", flush=True)

ALT_H = [torch.tensor(np.array(real[k]["activation_vector"], dtype=np.float32),
                      device=dev) for k in list(real)[:5]]
rows_out = []
for si, (key, t_ent) in enumerate(by_span.items()):
    if key not in real:
        continue
    r = real[key]
    h = torch.tensor(np.array(r["activation_vector"], dtype=np.float32), device=dev)
    v_local = torch.tensor(
        np.frombuffer(r["transported_vectors"], np.float16).reshape(16, -1)[:L]
        .astype(np.float32), device=dev).sum(0)
    span_ids = list(key)

    seqs, plist, lps = [], [], []
    for c in t_ent["contexts"]:
        cid = tok.encode(c, add_special_tokens=False)
        if len(cid) < 8:
            continue
        cid = cid[-480:]
        seqs.append(cid + span_ids)
        plist.append(len(cid) - 1)
    if len(seqs) < 4:
        continue
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    ids, mask = ids.to(dev), mask.to(dev)
    p_pos = torch.tensor(plist, device=dev)

    # P(S | generated context): does the model actually want to say S here?
    with torch.no_grad():
        lg = model(input_ids=ids, attention_mask=mask).logits.float().log_softmax(-1)
    for i, p in enumerate(plist):
        lp = sum(float(lg[i, p + k, span_ids[k]]) for k in range(L))
        lps.append(lp)

    tr = dvjp_transports(model, ids, mask, p_pos,
                        h.unsqueeze(0).expand(len(seqs), -1))[0]     # [B,16,d]
    per_ctx = tr[:, :L].float().sum(1)                               # pool d<L

    # Is J_S h a function of BOTH the span and the activation, or effectively a
    # per-span constant? If applying the same span-averaged operator to a foreign
    # activation gives nearly the same vector, then the training set is a codebook
    # of N discrete span vectors and the decoder becomes a lookup — which works for
    # enumerated spans (this is what the paper's template lens accepts) but cannot
    # interpolate to unseen ones. Low cosine here means h genuinely contributes.
    h_alt = ALT_H[si % len(ALT_H)]
    if float(cos(h_alt, h, dim=-1)) < 0.98:
        tr2 = dvjp_transports(model, ids, mask, p_pos,
                              h_alt.unsqueeze(0).expand(len(seqs), -1))[0]
        ent_alt = float(cos(per_ctx.mean(0), tr2[:, :L].float().sum(1).mean(0), dim=-1))
    else:
        ent_alt = float("nan")
    keep = [i for i, lp in enumerate(lps) if lp >= args.min_logprob]
    ent = {
        "span": t_ent["span_text"], "n_ctx": len(seqs), "n_kept": len(keep),
        "logprob_mean": float(np.mean(lps)), "logprob_max": float(np.max(lps)),
        "cos_global_vs_spanavg": float(cos(JP @ h, per_ctx.mean(0), dim=-1)),
        "cos_global_vs_local": float(cos(JP @ h, v_local, dim=-1)),
        "cos_spanavg_vs_local": float(cos(per_ctx.mean(0), v_local, dim=-1)),
        "cos_spanavg_own_h_vs_foreign_h": ent_alt,
    }
    if keep:
        ent["cos_global_vs_spanavg_filtered"] = float(
            cos(JP @ h, per_ctx[keep].mean(0), dim=-1))
    rows_out.append(ent)
    if si % 20 == 0:
        print(f"  [{si}] {t_ent['span_text']!r} kept={len(keep)}/{len(seqs)} "
              f"cos(global,spanavg)={ent['cos_global_vs_spanavg']:+.3f} "
              f"cos(global,local)={ent['cos_global_vs_local']:+.3f}", flush=True)

def mean(k):
    v = [r[k] for r in rows_out if k in r]
    return float(np.mean(v)) if v else float("nan")


res = {"n_spans": len(rows_out), "span_len": L, "per_span": rows_out,
       "mean_cos_global_vs_spanavg": mean("cos_global_vs_spanavg"),
       "mean_cos_global_vs_spanavg_filtered": mean("cos_global_vs_spanavg_filtered"),
       "mean_cos_global_vs_local": mean("cos_global_vs_local"),
       "mean_cos_spanavg_vs_local": mean("cos_spanavg_vs_local"),
       "mean_logprob": mean("logprob_mean"),
       "mean_h_sensitivity": mean("cos_spanavg_own_h_vs_foreign_h"),
       "mean_kept": mean("n_kept")}
print(f"\n{len(rows_out)} spans measured")
print(f"  mean total logprob of S under generated contexts   {res['mean_logprob']:+.2f}")
print(f"  contexts surviving the logprob filter              {res['mean_kept']:.1f}")
print(f"\n  cos(global Jbar h, SPAN-AVERAGED Jacobian h)      "
      f"{res['mean_cos_global_vs_spanavg']:+.4f}   <- this design")
print(f"  ... restricted to filtered contexts               "
      f"{res['mean_cos_global_vs_spanavg_filtered']:+.4f}")
print(f"  cos(global Jbar h, LOCAL Jacobian h)              "
      f"{res['mean_cos_global_vs_local']:+.4f}   <- arm I's mismatch")
hs = res["mean_h_sensitivity"]
print(f"\n  cos(J_S h_own, J_S h_foreign)                     {hs:+.4f}")
print("   high (>0.95) => J_S h is effectively a per-span CONSTANT, so the training"
      "\n   set is a codebook of discrete span vectors: fine for enumerated spans"
      "\n   (what the paper's template lens accepts) but cannot interpolate to unseen"
      "\n   ones. low => the activation genuinely contributes and interpolation is"
      "\n   possible.")
g = res["mean_cos_global_vs_spanavg"] - res["mean_cos_global_vs_local"]
res["improvement"] = g
print(f"\nmismatch reduction: {g:+.4f}")
print("VERDICT:", "span-averaging DOES sit closer to the global operator — worth "
      "building the full training set" if g > 0.05 else
      "span-averaging is NOT closer to the global operator than a single context is; "
      "the mismatch is not what this design fixes")
json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
print(f"wrote {args.out}", flush=True)
