"""Per-horizon quality of the test-time estimate, against the REAL state.

The central claim of this study is that the averaged-Jacobian estimate is good
enough to drive a lens at horizon 0 and not beyond. That claim needs the right
measurement: how close is Jbar^(d) @ h42[p] to the ACTUAL penultimate state
h62[p+d] that a decoder trained on real states expects?

Earlier numbers conflated two different comparisons — the audit measured
cos(Jbar^0 h42, h62[p]) = 0.50 at horizon 0 only, and the 0.10-0.16 figures were
against LOCAL TRANSPORTS, a different object. This measures estimate-vs-real
state at every horizon, on held-out rows, using the pen8 shards which store both
h42[p] (activation_vector) and the real h62[p..p+15] (transported_vectors).

Reports per horizon d:
  cos_raw          cos(Jbar^(d) h42, h62[p+d])
  cos_affine       cos(Jbar^(d) h42 + b_d, h62[p+d])     b_d = E[h62 - Jbar h42]
  cos_baseline     cos(h42[p], h62[p+d])                 does the transport help
                                                          at all over raw h42?
  norm_ratio       ||Jbar^(d) h42|| / ||h62[p+d]||
  cos_to_mean      cos(h62[p+d], mu_62)                  how much of the target
                                                          is just the mean
"""
import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="/workspace/data/spans_pen8/shard_3_jvp.parquet")
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--n-rows", type=int, default=2048)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--out", default="/workspace/results/multislot_eval/estimate_quality.json")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
K = args.k

Jb = []
for d in range(K):
    p = f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy"
    try:
        Jb.append(torch.from_numpy(np.load(p)).float().to(dev))
    except FileNotFoundError:
        break
K = len(Jb)
mu62 = torch.from_numpy(np.load(f"{args.jbar_dir}/hbar_L62.npy")).float().to(dev)
print(f"{K} per-offset matrices; ||mu_62||={float(mu62.norm()):.1f}", flush=True)

t = pq.read_table(sorted(glob.glob(args.shards))[0],
                  columns=["activation_vector", "transported_vectors"])
rows = t.slice(0, args.n_rows).to_pylist()
H = torch.tensor(np.array([r["activation_vector"] for r in rows], dtype=np.float32),
                 device=dev)                                        # h42[p]
S = torch.tensor(np.array([np.frombuffer(r["transported_vectors"], np.float16)
                           .reshape(16, -1)[:K] for r in rows], dtype=np.float32),
                 device=dev)                                        # real h62[p+d]
N = H.shape[0]
print(f"{N} rows | ||h42||={float(H.norm(dim=-1).mean()):.1f} "
      f"||h62||={float(S.norm(dim=-1).mean()):.1f}", flush=True)

cos = torch.nn.functional.cosine_similarity
res = {"n_rows": N, "per_horizon": []}
print("\n  d | cos(est,real) | +affine | cos(h42,real) | |est|/|real| | cos(real,mu62)")
for d in range(K):
    est = (Jb[d] @ H.T).T
    real = S[:, d]
    b = (real - est).mean(0)                       # the affine anchor for this d
    row = {
        "delta": d,
        "cos_raw": float(cos(est, real, dim=-1).mean()),
        "cos_affine": float(cos(est + b, real, dim=-1).mean()),
        "cos_baseline_h42": float(cos(H, real, dim=-1).mean()),
        "norm_ratio": float((est.norm(dim=-1) / real.norm(dim=-1)).mean()),
        "cos_real_to_mean": float(cos(real, mu62.expand_as(real), dim=-1).mean()),
        "b_norm": float(b.norm()),
    }
    res["per_horizon"].append(row)
    print(f" {d:2d} |    {row['cos_raw']:+.3f}     |  {row['cos_affine']:+.3f}  "
          f"|    {row['cos_baseline_h42']:+.3f}     |    {row['norm_ratio']:.3f}     "
          f"|    {row['cos_real_to_mean']:+.3f}", flush=True)

json.dump(res, open(args.out, "w"), indent=2)
print(f"\nwrote {args.out}", flush=True)
r0 = res["per_horizon"][0]
usable = [r["delta"] for r in res["per_horizon"] if r["cos_raw"] >= 0.9 * r0["cos_raw"]]
print(f"VERDICT: horizons within 10% of horizon-0 estimate quality: {usable}", flush=True)
