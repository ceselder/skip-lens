"""Pick the pooling weights under the metric that actually predicts the readout.

The first weight sweep (diag_pool_weights.py) scored candidates by cosine to the
local transport, and the horizon-matrix result then showed cosine to be the wrong
metric — it ranked the offset-0-dominated pools highest precisely because they
carry immediate-token content the raw activation already has.

So redo the sweep in readout space, before arm K spends 8 GPU-hours on a
weighting chosen under the wrong objective. For each candidate pooling w, decode
Σ w_d J̄⁽ᵈ⁾h through the unembedding and measure how often the tokens that
actually follow appear in the top-k, minus the same measurement against a
DIFFERENT row's continuation so that generic token frequency is subtracted.

Scored on the SPAN a decoder would have to emit (offsets 1-7) and on the long
tail (1-15), always excluding offset 0 — predicting the immediate token is not
what a multi-token workspace lens is for, and it is the one thing the raw
activation already does well.
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
ap.add_argument("--prefix", default="Jbar",
                help="Jbar for the plain averaged Jacobians, Wreg for the "
                     "ridge-regression maps from fit_regression_lens.py")
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--src-layer", type=int, default=42)
ap.add_argument("--tgt-layer", type=int, default=62)
ap.add_argument("--n-rows", type=int, default=512)
ap.add_argument("--topk", type=int, default=50)
ap.add_argument("--out", default="/workspace/results/multislot_eval/pool_readout.json")
args = ap.parse_args()
dev = "cuda"
K = 16

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
model = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)
inner = model.model if hasattr(model, "model") else model
NORM, WU = inner.norm, model.lm_head.weight

Jb = [torch.from_numpy(np.load(
    f"{args.jbar_dir}/{args.prefix}_L{args.src_layer}_to_L{args.tgt_layer}_off{d}.npy")
    ).float().to(dev) for d in range(K)]
nrm = [float(J.norm()) for J in Jb]

t = pq.read_table(sorted(glob.glob(args.shards))[0],
                  columns=["activation_vector", "rollout_token_ids",
                           "h42_recompute_cosine"])
rows = [r for r in t.slice(0, args.n_rows * 3).to_pylist()
        if r["h42_recompute_cosine"] >= 0.99 and len(r["rollout_token_ids"]) >= K
        ][: args.n_rows]
N = len(rows)
FUT = np.array([[int(x) for x in r["rollout_token_ids"][:K]] for r in rows])
H = torch.tensor(np.array([r["activation_vector"] for r in rows], dtype=np.float32),
                 device=dev)
print(f"{N} rows, top-{args.topk}, L{args.src_layer}->L{args.tgt_layer}", flush=True)

CANDS = {"raw_h42": None, "d=0 only (J-lens vector)": [1.0] + [0.0] * 15}
for lo in (1, 2, 3):
    for hi in (8, 16):
        CANDS[f"uniform {lo}<=d<{hi}"] = [1.0 if lo <= d < hi else 0.0 for d in range(K)]
        CANDS[f"norm-eq {lo}<=d<{hi}"] = [1.0 / nrm[d] if lo <= d < hi else 0.0
                                          for d in range(K)]
CANDS["uniform d<16"] = [1.0] * K
for d in (1, 2, 4, 8):
    CANDS[f"d={d} alone"] = [1.0 if e == d else 0.0 for e in range(K)]
# ramp: upweight deep offsets beyond norm equalisation, since the diffuse future
# tail is flat in d while the immediate term decays — a ramp should favour tail
CANDS["norm-eq x d, 1<=d<16"] = [d / nrm[d] if d >= 1 else 0.0 for d in range(K)]

SH = N // 2
out = []
for name, w in CANDS.items():
    JP = None if w is None else sum(w[d] * Jb[d] for d in range(K) if w[d])
    span, tail, span_s, tail_s = [], [], [], []
    for i in range(N):
        v = H[i] if JP is None else JP @ H[i]
        h = NORM(v.to(WU.dtype).unsqueeze(0))
        top = set((h @ WU.T).float()[0].topk(args.topk).indices.tolist())
        o = (i + SH) % N
        span.append(len(top & set(FUT[i, 1:8].tolist())) / 7)
        tail.append(len(top & set(FUT[i, 1:].tolist())) / 15)
        span_s.append(len(top & set(FUT[o, 1:8].tolist())) / 7)
        tail_s.append(len(top & set(FUT[o, 1:].tolist())) / 15)
    out.append({"weights": name,
                "span_1_7": float(np.mean(span)), "span_shuffled": float(np.mean(span_s)),
                "span_net": float(np.mean(span) - np.mean(span_s)),
                "tail_1_15": float(np.mean(tail)), "tail_shuffled": float(np.mean(tail_s)),
                "tail_net": float(np.mean(tail) - np.mean(tail_s)),
                "w": w})
    print(f"  {name:28s} span_net={out[-1]['span_net']:+.4f} "
          f"tail_net={out[-1]['tail_net']:+.4f}", flush=True)

out.sort(key=lambda r: -r["span_net"])
print(f"\n{'weights':28s}  span(1-7) shuffled   NET    tail(1-15)  NET")
for r in out:
    print(f"{r['weights']:28s}   {r['span_1_7']:.3f}    {r['span_shuffled']:.3f}   "
          f"{r['span_net']:+.4f}    {r['tail_1_15']:.3f}   {r['tail_net']:+.4f}")

best = out[0]
raw = next(r for r in out if r["weights"] == "raw_h42")
cur = next(r for r in out if r["weights"] == "norm-eq 1<=d<16")
print(f"\nBEST on the 8-token span: {best['weights']}  net {best['span_net']:+.4f}")
print(f"  raw activation control:            {raw['span_net']:+.4f}")
print(f"  what arm K is currently training:  {cur['span_net']:+.4f} "
      f"({'already optimal' if best['weights'] == cur['weights'] else 'REBUILD WORTH IT'})")
json.dump(out, open(args.out, "w"), indent=1)
print(f"wrote {args.out}", flush=True)
