"""Is the averaged Jacobian family horizon-RESOLVED, or just generically future?

This is the structural question the whole multi-slot design rests on, and it has
never been measured directly. J̄⁽ᵈ⁾ is fit as the average of ∂h62,t+d/∂h42,t, so
slot d is *supposed* to be about the token d ahead. If that is true the family is
horizon-resolved and a multi-slot format is the right idea with the right
readout. If J̄⁽ᵈ⁾ instead predicts all future tokens about equally, the offsets
carry one shared "future" signal, only pooling makes sense, and the eight-slot
design was doomed from the start regardless of format.

Measured in readout space, since the cosine metric already misled us once: for
each source offset d and each target offset e, the rate at which the token
actually at offset e appears in the top-k of J̄⁽ᵈ⁾h. Every cell is corrected by
the same measurement against a DIFFERENT row's continuation, which subtracts
generic token frequency — without that correction raw frequency dominates and
everything looks informative.

Diagonal dominance (cell (d,d) beating (d,e≠d)) means horizon-resolved. A flat
row means offset d knows "something future" but not which token.
"""
import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="/workspace/data/spans_jvp/shard_3_jvp.parquet")
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--n-rows", type=int, default=768)
ap.add_argument("--topk", type=int, default=50)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--src-layer", type=int, default=42)
ap.add_argument("--tgt-layer", type=int, default=62)
ap.add_argument("--out", default="/workspace/results/multislot_eval/horizon_matrix.json")
args = ap.parse_args()
dev = "cuda"
K = args.k

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
model = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)
inner = model.model if hasattr(model, "model") else model
NORM, WU = inner.norm, model.lm_head.weight

Jb = [torch.from_numpy(np.load(f"{args.jbar_dir}/Jbar_L{args.src_layer}_to_L{args.tgt_layer}_off{d}.npy")).float().to(dev)
      for d in range(K)]
print(f"{K} averaged per-offset matrices "
      f"(L{args.src_layer} -> L{args.tgt_layer})", flush=True)

t = pq.read_table(sorted(glob.glob(args.shards))[0],
                  columns=["activation_vector", "rollout_token_ids",
                           "h42_recompute_cosine"])
rows = [r for r in t.slice(0, args.n_rows * 3).to_pylist()
        if r["h42_recompute_cosine"] >= 0.99 and len(r["rollout_token_ids"]) >= K
        ][: args.n_rows]
N = len(rows)
print(f"{N} rows, top-{args.topk}", flush=True)

FUT = np.array([[int(x) for x in r["rollout_token_ids"][:K]] for r in rows])
H = torch.tensor(np.array([r["activation_vector"] for r in rows], dtype=np.float32),
                 device=dev)

hit = np.zeros((K, K))          # (source offset d, target offset e)
shuf = np.zeros((K, K))
SH = N // 2
for i in range(N):
    for d in range(K):
        h = NORM((Jb[d] @ H[i]).to(WU.dtype).unsqueeze(0))
        top = set((h @ WU.T).float()[0].topk(args.topk).indices.tolist())
        for e in range(K):
            hit[d, e] += FUT[i, e] in top
            shuf[d, e] += FUT[(i + SH) % N, e] in top
    if i % 150 == 0:
        print(f"  row {i}/{N}", flush=True)
hit /= N
shuf /= N
net = hit - shuf                 # frequency-corrected

print("\nfrequency-corrected recall, rows = source offset d, cols = target offset e")
print("      " + "".join(f"  e={e:<3d}" for e in range(0, K, 2)))
for d in range(K):
    print(f" d={d:2d} " + "".join(f" {net[d, e]:+.3f}" for e in range(0, K, 2)))

diag = float(np.mean([net[d, d] for d in range(K)]))
off = float(np.mean([net[d, e] for d in range(K) for e in range(K) if d != e]))
best_e = [int(np.argmax(net[d])) for d in range(K)]
on_diag = sum(1 for d in range(K) if best_e[d] == d)
row_peak_at_0 = sum(1 for d in range(K) if best_e[d] == 0)

res = {"n_rows": N, "topk": args.topk,
       "net": net.tolist(), "hit": hit.tolist(), "shuffled": shuf.tolist(),
       "mean_diagonal": diag, "mean_off_diagonal": off,
       "argmax_target_per_source": best_e,
       "n_sources_peaking_on_own_horizon": on_diag,
       "n_sources_peaking_at_offset_0": row_peak_at_0}

print(f"\n  mean on-diagonal      {diag:+.4f}")
print(f"  mean off-diagonal     {off:+.4f}")
print(f"  argmax target per source offset: {best_e}")
print(f"  sources peaking on their OWN horizon: {on_diag}/{K}")
print(f"  sources peaking at offset 0 instead:  {row_peak_at_0}/{K}")
print("\nVERDICT:", "HORIZON-RESOLVED — slot d really is about token d, so a "
      "multi-slot format is the right idea and the readout was the problem"
      if on_diag >= K // 2 else
      "NOT horizon-resolved — the offsets share one generic future signal, so "
      "only pooling is meaningful and per-offset slots could never have worked")
json.dump(res, open(args.out, "w"), indent=1)
print(f"wrote {args.out}", flush=True)
