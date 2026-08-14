"""Is arm K's training target actually norm-equalised? (No.)

Arm K's weights are w_d = 1/||Jbar^(d)||, derived from the AVERAGED matrices. But
in the training target those weights multiply the LOCAL transports
J_local^(d)(ctx) h, which have a different norm profile — the averaged transports
decay faster with d than the local ones do (||Jbar h||/||J_local h|| falls from
0.61 at d=0 to 0.26 at d=7).

So the weighting is exactly right for the test-time operator and WRONG for the
thing the decoder learned from: it over-amplifies deep local transports. If the
weighted local contributions are far from equal, arm K's underperformance may be a
mis-weighted training target rather than evidence against excluding offset 0.

Prints, per offset: the mean local-transport norm, the averaged matrix norm, arm
K's weight, the resulting weighted contribution (equal across d = well
conditioned), and the weight that WOULD have equalised the local transports.
"""
import argparse
import glob

import numpy as np
import pyarrow.parquet as pq
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="/workspace/data/spans_jvp/shard_3_jvp.parquet")
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--n-rows", type=int, default=3000)
ap.add_argument("--k", type=int, default=16)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
K = args.k

t = pq.read_table(sorted(glob.glob(args.shards))[0],
                  columns=["transported_vectors", "h42_recompute_cosine"])
rows = [r for r in t.slice(0, args.n_rows * 3).to_pylist()
        if r["h42_recompute_cosine"] >= 0.99][: args.n_rows]
V = torch.tensor(np.array([np.frombuffer(r["transported_vectors"], np.float16)
                           .reshape(16, -1)[:K] for r in rows], dtype=np.float32), device=dev)
ln = V.norm(dim=-1).mean(0).cpu().numpy()
jn = [float(np.linalg.norm(np.load(
    f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy", mmap_mode="r"))) for d in range(K)]
wK = [0.0] + [1.0 / jn[d] for d in range(1, K)]
wL = [0.0] + [1.0 / float(ln[d]) for d in range(1, K)]

print(f"{len(rows)} held-out rows\n")
print("  d   ||local v_d||   ||Jbar_d||   armK w_d   w_d*||v_d||   ideal w_d")
for d in range(K):
    print(f" {d:2d}    {ln[d]:9.2f}    {jn[d]:8.3f}   {wK[d]:8.4f}   "
          f"{wK[d] * ln[d]:9.3f}   {wL[d]:9.4f}")

c = [wK[d] * ln[d] for d in range(1, K)]
print(f"\nweighted local contributions span {max(c) / min(c):.2f}x "
      f"(1.00x = truly equalised)")
print(f"unweighted local transports span  {max(ln[1:]) / min(ln[1:]):.2f}x")
print(f"averaged matrices span            {max(jn[1:]) / min(jn[1:]):.1f}x")
big = int(np.argmax(c)) + 1
print(f"largest weighted contribution is offset {big} at {max(c):.3f}, "
      f"smallest is offset {int(np.argmin(c)) + 1} at {min(c):.3f}")
print("\nVERDICT:", "arm K's target IS roughly equalised — the weighting is not "
      "the problem" if max(c) / min(c) < 1.6 else
      "arm K's target is NOT equalised: the averaged-matrix weights mis-scale the "
      "local transports, so the decoder trained on a target dominated by the "
      "offsets where the two families disagree most")
