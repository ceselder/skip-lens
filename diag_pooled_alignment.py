"""Does token-POOLING rescue the train/test alignment that per-offset slots lost?

Arm I's whole bet is that pooling over token offsets makes the averaged Jacobian
a usable stand-in for the local one. That is a pure geometry question, answerable
from data already on disk in about a minute, with no model and no training — so
it should be asked before spending 7.6 GPU-hours.

For each held-out row, with h = h42[p]:

    train vector  v_local = sum_d  J_local^(d)(this context) @ h   (stored)
    test  vector  v_avg   = sum_d  Jbar^(d)                  @ h   (matrices)

and the number that matters is cos(v_avg, v_local). The per-offset slots measured
0.29 on this quantity, which is why nothing downstream worked. Pooling helps only
if the sum aligns much better than its terms do.

Two controls, because a high cosine can be vacuous:
  * cos to the corpus-mean pooled vector — if v_avg is mostly the shared mean,
    a high cosine says "both are near the mean", not "the transport is faithful".
    Reported both raw and after removing that mean from both sides.
  * cos(h, v_local) — raw h42 as the test vector instead of the transport. If
    raw h42 aligns with the training target just as well, the averaged Jacobian
    earns nothing, matching the thin 0.585-vs-0.521 margin seen at horizon 0.
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
ap.add_argument("--n-rows", type=int, default=4096)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--pool-upto", type=int, default=8,
                help="arm I trains on an 8-token span, so pool over d<8 by default")
ap.add_argument("--out", default="/workspace/results/multislot_eval/pooled_alignment.json")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
cos = torch.nn.functional.cosine_similarity

Jb = []
for d in range(args.k):
    try:
        Jb.append(torch.from_numpy(
            np.load(f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy")).float().to(dev))
    except FileNotFoundError:
        break
K = len(Jb)
print(f"{K} averaged per-offset matrices | pooling over d<{args.pool_upto}", flush=True)

t = pq.read_table(sorted(glob.glob(args.shards))[0],
                  columns=["activation_vector", "transported_vectors",
                           "h42_recompute_cosine"])
rows = [r for r in t.slice(0, args.n_rows * 2).to_pylist()
        if r["h42_recompute_cosine"] >= 0.99][: args.n_rows]
H = torch.tensor(np.array([r["activation_vector"] for r in rows], dtype=np.float32),
                 device=dev)
L = torch.tensor(np.array([np.frombuffer(r["transported_vectors"], np.float16)
                           .reshape(16, -1)[:K] for r in rows], dtype=np.float32),
                 device=dev)                                  # local transports
print(f"{H.shape[0]} rows | ||h42||={float(H.norm(dim=-1).mean()):.1f}", flush=True)

res = {"n_rows": H.shape[0], "pool_upto": args.pool_upto, "per_offset": [], "pooled": {}}

print("\nPER-OFFSET (what the eight-slot arms were asked to do)")
print("  d | cos(avg, local) | cos(h42, local) | |avg|/|local|")
for d in range(K):
    a, l = (Jb[d] @ H.T).T, L[:, d]
    row = {"delta": d,
           "cos_avg_local": float(cos(a, l, dim=-1).mean()),
           "cos_h42_local": float(cos(H, l, dim=-1).mean()),
           "norm_ratio": float((a.norm(dim=-1) / (l.norm(dim=-1) + 1e-9)).mean())}
    res["per_offset"].append(row)
    if d < args.pool_upto:
        print(f" {d:2d} |     {row['cos_avg_local']:+.3f}      |     "
              f"{row['cos_h42_local']:+.3f}      |    {row['norm_ratio']:7.3f}", flush=True)

P = args.pool_upto
JP = torch.zeros_like(Jb[0])
for d in range(min(P, K)):
    JP = JP + Jb[d]
v_avg = (JP @ H.T).T                       # test vector
v_loc = L[:, :P].sum(1)                    # train vector
mu_avg, mu_loc = v_avg.mean(0), v_loc.mean(0)

res["pooled"] = {
    "cos_avg_local": float(cos(v_avg, v_loc, dim=-1).mean()),
    "cos_avg_local_centered": float(cos(v_avg - mu_avg, v_loc - mu_loc, dim=-1).mean()),
    "cos_h42_local": float(cos(H, v_loc, dim=-1).mean()),
    "cos_avg_to_its_mean": float(cos(v_avg, mu_avg.expand_as(v_avg), dim=-1).mean()),
    "cos_local_to_its_mean": float(cos(v_loc, mu_loc.expand_as(v_loc), dim=-1).mean()),
    "norm_ratio": float((v_avg.norm(dim=-1) / v_loc.norm(dim=-1)).mean()),
    "mean_per_offset_cos": float(np.mean([r["cos_avg_local"]
                                          for r in res["per_offset"][:P]])),
}
p = res["pooled"]
print(f"\nPOOLED over d<{P} (arm I)")
print(f"  cos(v_avg, v_local)              {p['cos_avg_local']:+.3f}   "
      f"<- the train/test gap arm I must survive")
print(f"  ... mean removed from both sides  {p['cos_avg_local_centered']:+.3f}   "
      f"<- how much is NOT the shared mean")
print(f"  cos(raw h42, v_local)            {p['cos_h42_local']:+.3f}   "
      f"<- the control the Jacobian must beat")
print(f"  mean of the per-offset cosines    {p['mean_per_offset_cos']:+.3f}   "
      f"<- what the eight-slot arms got")
print(f"  cos(v_avg, its own corpus mean)   {p['cos_avg_to_its_mean']:+.3f}")
print(f"  ||v_avg|| / ||v_local||           {p['norm_ratio']:.3f}")

gain = p["cos_avg_local"] - p["mean_per_offset_cos"]
beats = p["cos_avg_local"] - p["cos_h42_local"]
print(f"\nVERDICT")
print(f"  pooling changes alignment by {gain:+.3f} vs per-offset slots")
print(f"  the averaged transport beats raw h42 by {beats:+.3f}"
      f"{'  <- IT EARNS NOTHING' if beats <= 0.02 else ''}")
res["verdict"] = {"pooling_gain_vs_per_offset": gain, "transport_minus_raw_h42": beats}
json.dump(res, open(args.out, "w"), indent=1)
print(f"\nwrote {args.out}", flush=True)
