"""What does context-averaging cost, in effective dimensions?

The horizon matrix showed that averaged per-offset Jacobians are not
horizon-resolved, and the readout sweep showed one deep offset is as good as any
pool of sixteen. Both point at the same underlying fact, which this measures
directly: the averaged family is nearly degenerate.

For the deep offsets (d>=1), compare the LOCAL transports J_local^(d)h — the
per-example objects the decoder trains on — against their context-averaged
counterparts J̄^(d)h, on identical rows:

  mean pairwise |cos| across offsets   how copy-like the offsets are
  participation ratio 1/Σp²            effective dimensionality of the set
  top-1 eigenvalue share               how much sits in a single direction

The local family is horizon-specific by construction. If averaging collapses it,
per-offset slots were never carrying per-position information and no rearrangement
of them could have worked — which is exactly the pattern of six failed rescues.
"""
import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="/workspace/data/spans_jvp/shard_3_jvp.parquet")
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--src-layer", type=int, default=42)
ap.add_argument("--tgt-layer", type=int, default=62)
ap.add_argument("--n-rows", type=int, default=1024)
ap.add_argument("--out", default="/workspace/results/multislot_eval/family_rank.json")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
K = 16
cos = torch.nn.functional.cosine_similarity

Jb = [torch.from_numpy(np.load(
    f"{args.jbar_dir}/Jbar_L{args.src_layer}_to_L{args.tgt_layer}_off{d}.npy")
    ).float().to(dev) for d in range(K)]
t = pq.read_table(sorted(glob.glob(args.shards))[0],
                  columns=["activation_vector", "transported_vectors",
                           "h42_recompute_cosine"])
rows = [r for r in t.slice(0, args.n_rows * 3).to_pylist()
        if r["h42_recompute_cosine"] >= 0.99][: args.n_rows]
H = torch.tensor(np.array([r["activation_vector"] for r in rows], dtype=np.float32),
                 device=dev)
LOC = torch.tensor(np.array([np.frombuffer(r["transported_vectors"], np.float16)
                             .reshape(K, -1) for r in rows], dtype=np.float32), device=dev)
AVG = torch.stack([(Jb[d] @ H.T).T for d in range(K)], 1)
print(f"{H.shape[0]} rows, {K} offsets, L{args.src_layer}->L{args.tgt_layer}", flush=True)


def collinearity(X, lo):
    v = [float(cos(X[:, i], X[:, j], dim=-1).mean())
         for i in range(lo, K) for j in range(i + 1, K)]
    return {"mean_abs_cos": float(np.mean(np.abs(v))), "min_cos": float(np.min(v)),
            "max_cos": float(np.max(v)), "n_pairs": len(v)}


def spectrum(X, lo):
    """participation ratio of the row-normalised second-moment spectrum: the
    effective number of directions the set of transports occupies"""
    M = X[:, lo:].reshape(-1, X.shape[-1])
    M = M / (M.norm(dim=-1, keepdim=True) + 1e-9)
    s = torch.linalg.svdvals(M.T @ M / M.shape[0]).double().cpu().numpy()
    p = s / s.sum()
    return {"participation_ratio": float(1.0 / np.sum(p ** 2)),
            "top1_share": float(p[0]), "top10_share": float(p[:10].sum()),
            "n_vectors": int(M.shape[0])}


res = {"n_rows": H.shape[0], "src_layer": args.src_layer, "tgt_layer": args.tgt_layer}
for lo, tag in ((0, "all_offsets"), (1, "deep_only_d_ge_1")):
    res[tag] = {}
    for X, kind in ((LOC, "local"), (AVG, "averaged")):
        res[tag][kind] = {**collinearity(X, lo), **spectrum(X, lo)}
    a, l = res[tag]["averaged"], res[tag]["local"]
    res[tag]["rank_collapse_factor"] = l["participation_ratio"] / a["participation_ratio"]
    print(f"\n{tag}")
    print(f"  local     |cos|={l['mean_abs_cos']:.3f}  participation ratio "
          f"{l['participation_ratio']:7.2f}  top-1 {l['top1_share']:.3f}")
    print(f"  averaged  |cos|={a['mean_abs_cos']:.3f}  participation ratio "
          f"{a['participation_ratio']:7.2f}  top-1 {a['top1_share']:.3f}")
    print(f"  --> context-averaging costs {res[tag]['rank_collapse_factor']:.1f}x "
          f"in effective dimensionality")

json.dump(res, open(args.out, "w"), indent=1)
print(f"\nwrote {args.out}", flush=True)
