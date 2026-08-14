"""Is the regression lens's gain real, or is it just predicting the mean better?

fit_regression_lens.py reports cos(W h, v) = 0.296 against 0.171 for the plain
averaged Jacobian. That comparison is not yet trustworthy. A fixed linear map can
raise raw cosine simply by outputting something closer to the corpus-mean
transport, and the targets share a large mean component — cos(real state, mu_62)
was already ~0.5 in earlier measurements. Ridge regression with a large lambda is
*especially* prone to this, since shrinking toward zero leaves the mean-ish
direction dominant.

So report, on held-out rows and for both operators:

  cos_raw        cos(W h, v)                       what the fit script reported
  cos_centered   cos(W h - mean(W h), v - mean(v)) context-specific content only
  cos_to_mean    cos(W h, mean(W h))               how mean-dominated the output is
  eff_rank       effective dimensions, raw and after removing the mean

The centered number is the honest one. If the regression gain survives centering,
it is real structure; if it collapses to the baseline, the gain was mean-fitting
and the 62x rank collapse stands.
"""
import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--eval-shard", default="/workspace/data/spans_jvp/shard_3_jvp.parquet")
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--reg-dir", default="/workspace/results/regression_lens")
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--pool-from", type=int, default=1)
ap.add_argument("--eval-rows", type=int, default=4096)
ap.add_argument("--out", default="/workspace/results/regression_lens/honest_eval.json")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
K = args.k
cos = torch.nn.functional.cosine_similarity

t = pq.read_table(sorted(glob.glob(args.eval_shard))[0],
                  columns=["activation_vector", "transported_vectors",
                           "h42_recompute_cosine"])
rows = [r for r in t.slice(0, args.eval_rows * 3).to_pylist()
        if r["h42_recompute_cosine"] >= 0.99][: args.eval_rows]
H = torch.tensor(np.array([r["activation_vector"] for r in rows], dtype=np.float32),
                 device=dev)
V = torch.tensor(np.array([np.frombuffer(r["transported_vectors"], np.float16)
                           .reshape(16, -1)[:K] for r in rows], dtype=np.float32), device=dev)
VP = V[:, args.pool_from:].sum(1)
print(f"{H.shape[0]} held-out rows", flush=True)


def eff_rank(X, center):
    M = X - X.mean(0, keepdim=True) if center else X
    M = M / (M.norm(dim=-1, keepdim=True) + 1e-9)
    s = torch.linalg.svdvals(M.T @ M / M.shape[0]).double().cpu().numpy()
    p = s / s.sum()
    return float(1.0 / np.sum(p ** 2))


def report(name, pooled_pred, per_offset_pred):
    mu_p, mu_t = pooled_pred.mean(0), VP.mean(0)
    deep = torch.cat([per_offset_pred[d] for d in range(args.pool_from, K)])
    ent = {
        "cos_pooled_raw": float(cos(pooled_pred, VP, dim=-1).mean()),
        "cos_pooled_centered": float(cos(pooled_pred - mu_p, VP - mu_t, dim=-1).mean()),
        "cos_pooled_to_own_mean": float(cos(pooled_pred, mu_p.expand_as(pooled_pred),
                                            dim=-1).mean()),
        "cos_target_to_own_mean": float(cos(VP, mu_t.expand_as(VP), dim=-1).mean()),
        "cos_per_offset_raw": [float(cos(per_offset_pred[d], V[:, d], dim=-1).mean())
                               for d in range(K)],
        "eff_rank_deep_raw": eff_rank(deep, False),
        "eff_rank_deep_centered": eff_rank(deep, True),
        "norm_ratio_pooled": float((pooled_pred.norm(dim=-1) / VP.norm(dim=-1)).mean()),
    }
    ent["cos_per_offset_deep_mean"] = float(np.mean(ent["cos_per_offset_raw"][args.pool_from:]))
    res[name] = ent
    print(f"\n{name}")
    print(f"  cos pooled, raw            {ent['cos_pooled_raw']:+.4f}")
    print(f"  cos pooled, CENTERED       {ent['cos_pooled_centered']:+.4f}   <- honest")
    print(f"  cos output to its own mean {ent['cos_pooled_to_own_mean']:+.4f}")
    print(f"  deep per-offset cos mean   {ent['cos_per_offset_deep_mean']:+.4f}")
    print(f"  eff rank deep, raw         {ent['eff_rank_deep_raw']:7.1f}")
    print(f"  eff rank deep, CENTERED    {ent['eff_rank_deep_centered']:7.1f}")
    print(f"  ||pred|| / ||target||      {ent['norm_ratio_pooled']:.3f}")
    return ent


res = {"n_rows": H.shape[0], "pool_from": args.pool_from}
res["target_cos_to_own_mean"] = float(cos(VP, VP.mean(0).expand_as(VP), dim=-1).mean())
print(f"the pooled TARGET itself sits at cos {res['target_cos_to_own_mean']:+.3f} "
      f"to its own mean\n(any predictor gets that much raw cosine for free)", flush=True)

Jb = [torch.from_numpy(np.load(
    f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy")).float().to(dev) for d in range(K)]
jb_off = {d: (Jb[d] @ H.T).T for d in range(K)}
jb_pool = sum(jb_off[d] for d in range(args.pool_from, K))
report("plain averaged Jacobian E[J]", jb_pool, jb_off)

try:
    Wr = [torch.from_numpy(np.load(
        f"{args.reg_dir}/Wreg_L42_to_L62_off{d}.npy")).float().to(dev) for d in range(K)]
    wr_off = {d: (Wr[d] @ H.T).T for d in range(K)}
    WP = torch.from_numpy(np.load(
        f"{args.reg_dir}/Wreg_L42_to_L62_pooled{args.pool_from}.npy")).float().to(dev)
    report("ridge regression W*", (WP @ H.T).T, wr_off)
    # the per-offset sum is a different object from the directly-fit pooled map
    report("ridge W*, summed per-offset", sum(wr_off[d] for d in range(args.pool_from, K)),
           wr_off)
except FileNotFoundError as e:
    print(f"\nregression matrices not found yet ({e}) — run fit_regression_lens.py "
          f"--save-matrices first")

if "ridge regression W*" in res:
    b, w = res["plain averaged Jacobian E[J]"], res["ridge regression W*"]
    dr = w["cos_pooled_raw"] - b["cos_pooled_raw"]
    dc = w["cos_pooled_centered"] - b["cos_pooled_centered"]
    res["gain_raw"], res["gain_centered"] = dr, dc
    print(f"\nregression gain: {dr:+.4f} raw, {dc:+.4f} CENTERED")
    print("VERDICT:", "REAL — the gain survives mean removal" if dc > 0.5 * dr
          else "MOSTLY MEAN-FITTING — most of the raw gain is a better mean, not "
               "better context-specific content")
json.dump(res, open(args.out, "w"), indent=1)
print(f"wrote {args.out}", flush=True)
