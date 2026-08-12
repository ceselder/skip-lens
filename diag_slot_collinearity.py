"""Why does per_offset underperform a single slot? Measure slot DIVERSITY.

Hypothesis: the averaged transports J̄⁽ᐞ⁾h (test time) are mutually
near-collinear, while the local transports J⁽ᐞ⁾_local·h (train time) are
diverse. If so, the decoder is handed 8 near-copies of one vector at test
time — out of distribution in exactly the direction that makes it emit
generic filler, and consistent with per_offset ≈ pooled_identical.

Prints, over real rows: mean pairwise cosine among the K slots for
(a) stored LOCAL transports, (b) AVERAGED transports of the same h42,
plus per-slot cosine to slot 0 and effective rank of the slot matrix.
Also reports the same for two candidate FIXES:
  diff   : slot d -> (J̄⁽ᵈ⁾ - J̄⁽ᵈ⁻¹⁾) h   (differential content)
  gs     : Gram-Schmidt orthogonalization across slots
"""
import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--jvp-glob", default="/workspace/data/spans_jvp/shard_0_jvp.parquet")
ap.add_argument("--n-rows", type=int, default=256)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--out", default="/workspace/results/multislot_eval/slot_collinearity.json")
args = ap.parse_args()
K = args.k
dev = "cuda" if torch.cuda.is_available() else "cpu"

Jbar = torch.stack([torch.from_numpy(np.load(
    f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy")).float() for d in range(K)]).to(dev)
Jpool = torch.from_numpy(np.load(
    f"{args.jbar_dir}/Jbar_L42_to_L62_offpooled.npy")).float().to(dev)

f = sorted(glob.glob(args.jvp_glob))[0]
t = pq.read_table(f, columns=["activation_vector", "transported_vectors"])
rows = t.slice(0, args.n_rows).to_pylist()
print(f"{len(rows)} rows from {f}", flush=True)


def slot_stats(S):
    """S: [K, d] slot matrix -> (mean pairwise cos, cos-to-slot0 list, eff rank)."""
    U = torch.nn.functional.normalize(S, dim=-1)
    C = U @ U.T
    iu = torch.triu_indices(K, K, offset=1)
    mean_pair = float(C[iu[0], iu[1]].mean())
    to0 = [float(x) for x in C[0]]
    sv = torch.linalg.svdvals(S.float())
    er = float(((sv**2).sum() ** 2) / ((sv**4).sum()))  # participation ratio
    return mean_pair, to0, er


acc = {k: {"pair": [], "er": [], "to0": []} for k in ("local", "avg", "diff", "gs")}
for r in rows:
    h = torch.tensor(r["activation_vector"], dtype=torch.float32, device=dev)
    loc = torch.from_numpy(
        np.frombuffer(r["transported_vectors"], dtype=np.float16)
        .reshape(16, -1)[:K].astype(np.float32)).to(dev)
    avg = torch.einsum("kij,j->ki", Jbar, h)
    diff = torch.stack([avg[0]] + [avg[d] - avg[d - 1] for d in range(1, K)])
    q, _ = torch.linalg.qr(avg.T)          # columns = orthonormal basis of slot span
    gs = q.T[:K] * avg.norm(dim=-1, keepdim=True)
    for name, S in (("local", loc), ("avg", avg), ("diff", diff), ("gs", gs)):
        p, to0, er = slot_stats(S)
        acc[name]["pair"].append(p)
        acc[name]["er"].append(er)
        acc[name]["to0"].append(to0)

res = {}
for name, v in acc.items():
    res[name] = {
        "mean_pairwise_cos": float(np.mean(v["pair"])),
        "eff_rank_participation": float(np.mean(v["er"])),
        "cos_to_slot0_by_delta": [float(x) for x in np.mean(np.array(v["to0"]), axis=0)],
    }
    print(f"\n{name:6s} mean pairwise cos = {res[name]['mean_pairwise_cos']:+.3f} | "
          f"eff rank {res[name]['eff_rank_participation']:.2f} / {K}")
    print("       cos to slot0 by delta:",
          " ".join(f"{x:+.2f}" for x in res[name]["cos_to_slot0_by_delta"]))

json.dump(res, open(args.out, "w"), indent=2)
print(f"\nwrote {args.out}")
verdict = ("COLLINEAR — averaged slots carry ~1 direction; per_offset is "
           "effectively pooled_identical" if res["avg"]["mean_pairwise_cos"] > 0.7
           else "NOT collinear — look elsewhere")
print("VERDICT:", verdict)
