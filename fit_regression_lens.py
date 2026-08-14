"""Fit the OPTIMAL fixed linear map h42 -> transport, instead of averaging Jacobians.

The J-lens builds its readout as J̄ = E[J], the plain average of per-example
Jacobians. But E[J] is not the best fixed linear map from h to Jh. That is the
regression solution

    W* = E[(Jh) hᵀ] · E[h hᵀ]⁻¹

and the two coincide only when J is uncorrelated with h. They cannot be: h
encodes the context and J is that context's Jacobian, so they are strongly
dependent. Plain averaging discards exactly that dependence — which is a concrete
candidate mechanism for the 62x effective-rank collapse measured in
diag_family_rank.py (751 local dimensions -> 12 averaged ones).

W* is still a single fixed corpus-level matrix applied at test time, so it keeps
the property the design depends on: it cannot know this context's particular
continuation, only the corpus-level regularity, which is what turns "particular
use" into "general disposition to verbalize". It is the optimal linear
approximation to the averaging, not a per-example object.

Fit per offset by ridge regression over held-out-disjoint shards, in fp64
accumulators, then report against the plain averaged Jacobian on the same rows:

    cos(W_d h, v_d)   vs   cos(J̄⁽ᵈ⁾ h, v_d)      alignment to the real transport
    effective rank of {W_d h}  vs  {J̄⁽ᵈ⁾ h}       did the collapse reverse

Also fits the pooled target directly (a single W for the summed transport), since
the readout sweep found one vector is all the family carries.
"""
import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--fit-shards", default="/workspace/data/spans_jvp/shard_[0124]_jvp.parquet")
ap.add_argument("--eval-shard", default="/workspace/data/spans_jvp/shard_3_jvp.parquet")
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--max-fit-rows", type=int, default=160000)
ap.add_argument("--eval-rows", type=int, default=2048)
ap.add_argument("--lambdas", default="1e2,1e3,1e4,1e5")
ap.add_argument("--pool-from", type=int, default=1,
                help="the pooled target sums local transports from this offset up; "
                     "1 excludes the immediate-token term that raw h42 already carries")
ap.add_argument("--out-dir", default="/workspace/results/regression_lens")
ap.add_argument("--save-matrices", action="store_true")
args = ap.parse_args()
dev = "cuda"
K, D = args.k, 5120
os.makedirs(args.out_dir, exist_ok=True)
cos = torch.nn.functional.cosine_similarity

files = sorted(f for f in glob.glob(args.fit_shards) if not f.endswith("_probe.parquet"))
assert files, f"no fit shards matched {args.fit_shards}"
print(f"fitting on {len(files)} shards, holding out {os.path.basename(args.eval_shard)}",
      flush=True)

# fp64 accumulators: A = sum h h^T, Bd = sum v_d h^T, plus the pooled target
A = torch.zeros(D, D, dtype=torch.float64, device=dev)
B = torch.zeros(K, D, D, dtype=torch.float64, device=dev)
BP = torch.zeros(D, D, dtype=torch.float64, device=dev)
n = 0
for f in files:
    pf = pq.ParquetFile(f)
    for rg in range(pf.num_row_groups):
        if n >= args.max_fit_rows:
            break
        t = pf.read_row_group(rg, columns=["activation_vector", "transported_vectors",
                                           "h42_recompute_cosine"]).to_pylist()
        t = [r for r in t if r["h42_recompute_cosine"] >= 0.99]
        if not t:
            continue
        h = torch.tensor(np.array([r["activation_vector"] for r in t], dtype=np.float32),
                         device=dev).double()
        v = torch.tensor(np.array([np.frombuffer(r["transported_vectors"], np.float16)
                                   .reshape(16, -1)[:K] for r in t], dtype=np.float32),
                         device=dev).double()
        A += h.T @ h
        for d in range(K):
            B[d] += v[:, d].T @ h
        BP += v[:, args.pool_from:].sum(1).T @ h
        n += h.shape[0]
    print(f"  {os.path.basename(f)}: {n} rows", flush=True)
    if n >= args.max_fit_rows:
        break
print(f"accumulated {n} rows", flush=True)

# held-out evaluation rows
te = pq.read_table(sorted(glob.glob(args.eval_shard))[0],
                   columns=["activation_vector", "transported_vectors",
                            "h42_recompute_cosine"])
er = [r for r in te.slice(0, args.eval_rows * 3).to_pylist()
      if r["h42_recompute_cosine"] >= 0.99][: args.eval_rows]
He = torch.tensor(np.array([r["activation_vector"] for r in er], dtype=np.float32),
                  device=dev)
Ve = torch.tensor(np.array([np.frombuffer(r["transported_vectors"], np.float16)
                            .reshape(16, -1)[:K] for r in er], dtype=np.float32), device=dev)
VeP = Ve[:, args.pool_from:].sum(1)
Jb = [torch.from_numpy(np.load(
    f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy")).float().to(dev) for d in range(K)]
print(f"{He.shape[0]} held-out rows", flush=True)


def eff_rank(X):
    M = X / (X.norm(dim=-1, keepdim=True) + 1e-9)
    s = torch.linalg.svdvals(M.T @ M / M.shape[0]).double().cpu().numpy()
    p = s / s.sum()
    return float(1.0 / np.sum(p ** 2))


res = {"n_fit_rows": n, "n_eval_rows": He.shape[0], "pool_from": args.pool_from,
       "baseline": {}, "by_lambda": {}}

# baseline: the plain averaged Jacobians
base_pool = None
bcos = []
for d in range(K):
    a = (Jb[d] @ He.T).T
    bcos.append(float(cos(a, Ve[:, d], dim=-1).mean()))
    if d >= args.pool_from:
        base_pool = a if base_pool is None else base_pool + a
res["baseline"] = {
    "cos_per_offset": bcos,
    "cos_deep_mean": float(np.mean(bcos[args.pool_from:])),
    "cos_pooled": float(cos(base_pool, VeP, dim=-1).mean()),
    "eff_rank_deep": eff_rank(torch.cat([(Jb[d] @ He.T).T
                                         for d in range(args.pool_from, K)])),
}
print(f"\nBASELINE plain averaged Jacobian:")
print(f"  mean cos to local transport, deep offsets  {res['baseline']['cos_deep_mean']:+.4f}")
print(f"  cos on the pooled target                   {res['baseline']['cos_pooled']:+.4f}")
print(f"  effective rank of the deep transports      {res['baseline']['eff_rank_deep']:.1f}")

eye = torch.eye(D, dtype=torch.float64, device=dev)
best = (None, -1)
for lam in [float(x) for x in args.lambdas.split(",")]:
    Ainv = torch.linalg.inv(A + lam * eye)
    wcos, Wpool_cos = [], None
    deep_stack = []
    for d in range(K):
        W = (B[d] @ Ainv).float()
        a = (W @ He.T).T
        wcos.append(float(cos(a, Ve[:, d], dim=-1).mean()))
        if d >= args.pool_from:
            deep_stack.append(a)
        del W
    WP = (BP @ Ainv).float()
    Wpool_cos = float(cos((WP @ He.T).T, VeP, dim=-1).mean())
    ent = {"cos_per_offset": wcos,
           "cos_deep_mean": float(np.mean(wcos[args.pool_from:])),
           "cos_pooled": Wpool_cos,
           "eff_rank_deep": eff_rank(torch.cat(deep_stack))}
    res["by_lambda"][f"{lam:g}"] = ent
    print(f"\nRIDGE lambda={lam:g}")
    print(f"  mean cos to local transport, deep offsets  {ent['cos_deep_mean']:+.4f}  "
          f"({ent['cos_deep_mean'] - res['baseline']['cos_deep_mean']:+.4f} vs baseline)")
    print(f"  cos on the pooled target                   {ent['cos_pooled']:+.4f}  "
          f"({ent['cos_pooled'] - res['baseline']['cos_pooled']:+.4f} vs baseline)")
    print(f"  effective rank of the deep transports      {ent['eff_rank_deep']:.1f}  "
          f"(baseline {res['baseline']['eff_rank_deep']:.1f})")
    if ent["cos_pooled"] > best[1]:
        best = (lam, ent["cos_pooled"])
    del Ainv, WP, deep_stack

res["best_lambda"] = best[0]
print(f"\nBEST lambda {best[0]:g} on the pooled target: {best[1]:+.4f} "
      f"vs {res['baseline']['cos_pooled']:+.4f} for the plain average")
if args.save_matrices and best[0] is not None:
    Ainv = torch.linalg.inv(A + best[0] * eye)
    for d in range(K):
        np.save(f"{args.out_dir}/Wreg_L42_to_L62_off{d}.npy",
                (B[d] @ Ainv).float().cpu().numpy())
    np.save(f"{args.out_dir}/Wreg_L42_to_L62_pooled{args.pool_from}.npy",
            (BP @ Ainv).float().cpu().numpy())
    print(f"saved matrices to {args.out_dir}")
json.dump(res, open(f"{args.out_dir}/fit_report.json", "w"), indent=1)
print(f"wrote {args.out_dir}/fit_report.json", flush=True)
