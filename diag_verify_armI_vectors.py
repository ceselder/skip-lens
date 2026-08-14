"""Is arm I's +0.197 for the regression operator real, or an eval bug?

The judged gap (regression_avg 0.539 vs pooled_avg 0.342) is large enough to be
suspicious, and the supporting alignment numbers were measured for the DEEP pool
(offsets 1-15), not for the uniform d<16 pool that arm I actually trains and is
evaluated on. So the two things have never been compared on arm I's own object.

This rebuilds BOTH test-time vectors exactly as evals/pooled_single_eval.py does
for arm I (deep_from=0, weight_mode=uniform, i.e. an unweighted sum over all 16
offsets) and compares them against arm I's actual TRAINING target, which is the
unweighted sum of the 16 stored local transports for the same row.

If the regression operator is not markedly better here, the judged gap does not
come from a better test-time vector and something in the eval is wrong.

Also checks, because these are the ways this could silently break:
  * the pooled sum vs the standalone Jbar_..._offpooled.npy file (the paper's own
    pooled J-lens) — if those disagree, "pooled_avg" is not the paper's vector
  * norm ratio against the target, since a vector at the wrong scale gets
    norm-matched by the injection hook and can look fine in cosine yet not in a
    readout
  * cos to the corpus mean, to make sure neither is just predicting the mean
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
ap.add_argument("--reg-dir", default="/workspace/results/regression_lens")
ap.add_argument("--n-offsets", type=int, default=16)
ap.add_argument("--n-rows", type=int, default=4096)
ap.add_argument("--out", default="/workspace/results/regression_lens/verify_armI.json")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
K = args.n_offsets
cos = torch.nn.functional.cosine_similarity


def pooled(directory, prefix):
    acc = None
    for d in range(K):
        M = torch.from_numpy(np.load(
            f"{directory}/{prefix}_L42_to_L62_off{d}.npy")).float().to(dev)
        acc = M if acc is None else acc + M
    return acc


JP = pooled(args.jbar_dir, "Jbar")          # what pooled_avg feeds
WP = pooled(args.reg_dir, "Wreg")           # what regression_avg feeds
WPU = pooled(args.reg_dir, "WregU")
JPOOL_FILE = torch.from_numpy(np.load(
    f"{args.jbar_dir}/Jbar_L42_to_L62_offpooled.npy")).float().to(dev)
print(f"|sum_d Jbar_d| = {float(JP.norm()):.2f}   "
      f"|Jbar_offpooled file| = {float(JPOOL_FILE.norm()):.2f}   "
      f"cos = {float(cos(JP.flatten(), JPOOL_FILE.flatten(), dim=0)):.4f}")
print(f"|sum_d Wreg_d| = {float(WP.norm()):.2f}   |sum_d WregU_d| = {float(WPU.norm()):.2f}")

t = pq.read_table(sorted(glob.glob(args.shards))[0],
                  columns=["activation_vector", "transported_vectors",
                           "h42_recompute_cosine"])
rows = [r for r in t.slice(0, args.n_rows * 3).to_pylist()
        if r["h42_recompute_cosine"] >= 0.99][: args.n_rows]
H = torch.tensor(np.array([r["activation_vector"] for r in rows], dtype=np.float32),
                 device=dev)
V = torch.tensor(np.array([np.frombuffer(r["transported_vectors"], np.float16)
                           .reshape(16, -1)[:K] for r in rows], dtype=np.float32), device=dev)
TGT = V.sum(1)                       # arm I's training target, verbatim
print(f"\n{H.shape[0]} held-out rows | ||target||={float(TGT.norm(dim=-1).mean()):.1f}",
      flush=True)

res = {"n_rows": H.shape[0], "n_offsets": K,
       "cos_sumJbar_vs_offpooled_file": float(
           cos(JP.flatten(), JPOOL_FILE.flatten(), dim=0)),
       "conditions": {}}
mu_t = TGT.mean(0)
print(f"\n{'test vector':34s} cos(raw) cos(CENTERED) |pred|/|tgt| cos-to-own-mean")
for name, M in (("pooled_avg   = sum_d Jbar_d", JP),
                ("regression_avg = sum_d W*_d", WP),
                ("regrU        = sum_d W*U_d", WPU),
                ("jlens_pooled = offpooled file", JPOOL_FILE),
                ("raw_h42 (no operator)", None)):
    P = H if M is None else (M @ H.T).T
    mu_p = P.mean(0)
    ent = {"cos_raw": float(cos(P, TGT, dim=-1).mean()),
           "cos_centered": float(cos(P - mu_p, TGT - mu_t, dim=-1).mean()),
           "norm_ratio": float((P.norm(dim=-1) / TGT.norm(dim=-1)).mean()),
           "cos_to_own_mean": float(cos(P, mu_p.expand_as(P), dim=-1).mean())}
    res["conditions"][name] = ent
    print(f"{name:34s}  {ent['cos_raw']:+.4f}   {ent['cos_centered']:+.4f}    "
          f"{ent['norm_ratio']:7.3f}      {ent['cos_to_own_mean']:+.3f}")

c = res["conditions"]
g = (c["regression_avg = sum_d W*_d"]["cos_centered"]
     - c["pooled_avg   = sum_d Jbar_d"]["cos_centered"])
res["regression_minus_pooled_centered"] = g
print(f"\nregression - pooled, centered: {g:+.4f}")
print("VERDICT:", "consistent with the judged gap — the regression vector really is "
      "a much better match to what arm I trained on" if g > 0.05 else
      "NOT consistent — the two vectors are similarly aligned to the training "
      "target, so a 0.20 judged gap cannot come from vector quality. Suspect the "
      "eval.")
json.dump(res, open(args.out, "w"), indent=1)
print(f"wrote {args.out}", flush=True)
