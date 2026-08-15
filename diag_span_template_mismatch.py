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
# eager attention is REQUIRED, not a preference: the double-VJP is
# reverse-over-reverse, and aten::_scaled_dot_product_efficient_attention_backward
# has no derivative, so SDPA cannot be differentiated twice. The original pass-2
# collection loads eager for the same reason.
model = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

JP = None
for d in range(L):
    M = torch.from_numpy(np.load(
        f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy")).float().to(dev)
    JP = M if JP is None else JP + M
HBAR = torch.from_numpy(np.load(f"{args.jbar_dir}/hbar_L42.npy")).float().to(dev)
print(f"global Jbar pooled over d<{L}: |JP|={float(JP.norm()):.2f}", flush=True)

tpl = [json.loads(x) for x in open(args.templates)][: args.max_spans]
by_span = {tuple(t["span_ids"]): t for t in tpl}
print(f"{len(tpl)} spans with templates", flush=True)

# real held-out rows: give us h42 and the stored local transport for the SAME span
t = pq.read_table(args.shards, columns=["rollout_token_ids", "activation_vector",
                                        "transported_vectors", "h42_recompute_cosine",
                                        "ctx_text"])
real = {}
for r in t.slice(0, 60000).to_pylist():
    if r["h42_recompute_cosine"] < 0.99 or not r["rollout_token_ids"]:
        continue
    k = tuple(int(x) for x in r["rollout_token_ids"][:L])
    if k in by_span and k not in real:
        real[k] = r
print(f"{len(real)} spans matched to a real held-out row", flush=True)

def transports_for(ctx_list, span_ids, tangent, n_max=16):
    """mean over contexts of J_local(ctx) @ tangent, pooled over d < L.

    Every context gets the SAME span appended, so the only thing that differs
    between the span-matched and random-context sets is whether the context makes
    the span natural — which is what isolates span-conditioning from the mere
    sample-size effect of averaging over many contexts.
    """
    seqs, plist = [], []
    for c in ctx_list[:n_max]:
        cid = tok.encode(c, add_special_tokens=False)
        if len(cid) < 8:
            continue
        cid = cid[-480:]
        seqs.append(cid + span_ids + [PAD] * N_OFFSETS)
        plist.append(len(cid) - 1)
    if len(seqs) < 4:
        return None, None, None
    T = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    ids, mask = ids.to(dev), mask.to(dev)
    pp = torch.tensor(plist, device=dev)
    with torch.no_grad():
        lg = model(input_ids=ids, attention_mask=mask).logits.float().log_softmax(-1)
    lps = [sum(float(lg[i, p + k, span_ids[k]]) for k in range(L))
           for i, p in enumerate(plist)]
    tr = dvjp_transports(model, ids, mask, pp,
                         tangent.unsqueeze(0).expand(len(seqs), -1))[0]
    return tr[:, :L].float().sum(1), lps, (ids, mask, pp)


ALT_H = [torch.tensor(np.array(real[k]["activation_vector"], dtype=np.float32),
                      device=dev) for k in list(real)[:5]]
rows_out = []
VEC_GLOBAL, VEC_SPANAVG, VEC_LOCAL, VEC_RANDAVG = [], [], [], []
# (2) random-context control: contexts drawn from OTHER spans. If averaging over
# any 16 contexts lands as close to the global operator as span-matched ones do,
# the alignment gain is a sample-size effect and span-conditioning buys only label
# validity, not proximity.
ALL_CTX = [(k, c) for k, t in by_span.items() for c in t["contexts"]]
for si, (key, t_ent) in enumerate(by_span.items()):
    if key not in real:
        continue
    r = real[key]
    h = torch.tensor(np.array(r["activation_vector"], dtype=np.float32), device=dev)
    v_local = torch.tensor(
        np.frombuffer(r["transported_vectors"], np.float16).reshape(16, -1)[:L]
        .astype(np.float32), device=dev).sum(0)
    span_ids = list(key)

    # (a) span-matched contexts
    per_ctx, lps, batch = transports_for(t_ent["contexts"], span_ids, h)
    if per_ctx is None:
        continue

    # (b) RANDOM-context control: foreign contexts, same span appended. If this
    # lands as close to the global operator as (a) does, the alignment gain is a
    # sample-size effect and span-conditioning buys only label validity.
    foreign = [c for k, c in ALL_CTX if k != key]
    rnd = [foreign[(si * 7 + i * 13) % len(foreign)] for i in range(len(t_ent["contexts"]))]
    per_rnd, lps_rnd, _ = transports_for(rnd, span_ids, h)

    # (c) the REAL context the span actually came from, so -18 is interpretable:
    # the rollout was sampled at temperature 1.0 / top_p 0.95, so P(S) is not high
    # even where the model genuinely produced S. ctx_text is the last ~48 tokens.
    with torch.no_grad():
        rid = tok.encode(r["ctx_text"], add_special_tokens=False)[-480:]
        rseq = torch.tensor([rid + span_ids], device=dev)
        rlg = model(input_ids=rseq,
                    attention_mask=torch.ones_like(rseq)).logits.float().log_softmax(-1)
        lp_real = sum(float(rlg[0, len(rid) - 1 + k, span_ids[k]]) for k in range(L))

    # h-sensitivity, on CENTERED activations. Raw L42 activations are all >0.98
    # cosine to each other, so any linear operator maps them to near-parallel
    # outputs and the raw version of this test reads 1.0000 regardless of what the
    # operator does. Removing the corpus mean leaves the part that actually carries
    # context, which is what the decoder would have to rely on.
    h_alt = ALT_H[(si + 1) % len(ALT_H)]
    per_c, _, _ = transports_for(t_ent["contexts"], span_ids, h - HBAR)
    per_alt, _, _ = transports_for(t_ent["contexts"], span_ids, h_alt - HBAR)
    ent_alt = (float(cos(per_c.mean(0), per_alt.mean(0), dim=-1))
               if (per_alt is not None and per_c is not None) else float("nan"))

    keep = [i for i, lp in enumerate(lps) if lp >= args.min_logprob]
    ent = {
        "span": t_ent["span_text"], "n_ctx": int(per_ctx.shape[0]), "n_kept": len(keep),
        "logprob_mean": float(np.mean(lps)), "logprob_max": float(np.max(lps)),
        "cos_global_vs_spanavg": float(cos(JP @ h, per_ctx.mean(0), dim=-1)),
        "cos_global_vs_local": float(cos(JP @ h, v_local, dim=-1)),
        "cos_spanavg_vs_local": float(cos(per_ctx.mean(0), v_local, dim=-1)),
        "cos_spanavg_own_h_vs_foreign_h": ent_alt,
        "logprob_real_ctx": lp_real,
        "cos_global_vs_randavg": (float(cos(JP @ h, per_rnd.mean(0), dim=-1))
                                  if per_rnd is not None else float("nan")),
    }
    if keep:
        ent["cos_global_vs_spanavg_filtered"] = float(
            cos(JP @ h, per_ctx[keep].mean(0), dim=-1))
    if per_rnd is not None:
        VEC_RANDAVG.append(per_rnd.mean(0))
    VEC_GLOBAL.append(JP @ h)
    VEC_SPANAVG.append(per_ctx.mean(0))
    VEC_LOCAL.append(v_local)
    rows_out.append(ent)
    if si % 20 == 0:
        print(f"  [{si}] {t_ent['span_text']!r} kept={len(keep)}/{per_ctx.shape[0]} "
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
       "mean_cos_global_vs_randavg": mean("cos_global_vs_randavg"),
       "mean_logprob_real_ctx": mean("logprob_real_ctx"),
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
G = torch.stack(VEC_GLOBAL); S_ = torch.stack(VEC_SPANAVG); Lo = torch.stack(VEC_LOCAL)
res["centered_cos_global_vs_spanavg"] = float(
    cos(G - G.mean(0), S_ - S_.mean(0), dim=-1).mean())
res["centered_cos_global_vs_local"] = float(
    cos(G - G.mean(0), Lo - Lo.mean(0), dim=-1).mean())
print(f"\n  CENTERED cos(global, span-averaged)               "
      f"{res['centered_cos_global_vs_spanavg']:+.4f}   <- honest")
print(f"  CENTERED cos(global, local)                       "
      f"{res['centered_cos_global_vs_local']:+.4f}")
print("   raw cosines are inflated: any two L42 activations are >0.98 cosine, so"
      "\n   the shared mean dominates unless it is removed.")
print(f"\n  cos(global, RANDOM-context average)              "
      f"{res['mean_cos_global_vs_randavg']:+.4f}   <- sample-size control")
print(f"   if this matches the span-averaged number, the gain is sample size and"
      f"\n   span-conditioning buys label validity rather than proximity.")
print(f"\n  logprob of S under Claude's contexts              {res['mean_logprob']:+.2f}")
print(f"  logprob of S under the REAL context               "
      f"{res['mean_logprob_real_ctx']:+.2f}   <- the reference")
print(f"   the rollout was sampled at temperature 1.0 / top_p 0.95, so P(S) is not"
      f"\n   high even where the model genuinely produced S. Claude's contexts are only"
      f"\n   'weak' relative to this number, not in absolute terms.")
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
