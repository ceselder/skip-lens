"""Which pooling weights make the averaged Jacobian worth applying at all?

diag_pooled_alignment.py found that uniform pooling over d<8 aligns with the
training vector at 0.366 while raw h42 alone reaches 0.381 — the averaged
Jacobian earns nothing. The per-offset breakdown says why: at d=0 raw h42 is
already the better predictor (0.437 vs 0.395), and ‖J̄⁽⁰⁾‖ = 30.7 against 6.6,
3.4, … for the deeper offsets, so a uniform sum is dominated by the single
offset where the Jacobian has no edge. Every offset d≥1 favours the Jacobian.

So sweep the pooling weights w_d. Both sides of the arm use the SAME w — the
train vector is Σ w_d J_local⁽ᵈ⁾h and the test vector is Σ w_d J̄⁽ᵈ⁾h — so w is
a free design choice, not a train/test mismatch. For each candidate, report the
alignment and, decisively, how much it beats feeding raw h42.

Candidates: uniform over a range (with and without the dominant d=0), norm
equalising (w_d = 1/‖J̄⁽ᵈ⁾‖, so every horizon contributes comparably), and a mild
geometric taper as a middle ground.
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
ap.add_argument("--out", default="/workspace/results/multislot_eval/pool_weights.json")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
cos = torch.nn.functional.cosine_similarity

Jb = []
for d in range(16):
    try:
        Jb.append(torch.from_numpy(
            np.load(f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy")).float().to(dev))
    except FileNotFoundError:
        break
K = len(Jb)
nrm = [float(J.norm()) for J in Jb]
print(f"{K} matrices | ||Jbar|| = " + " ".join(f"{x:.1f}" for x in nrm), flush=True)

t = pq.read_table(sorted(glob.glob(args.shards))[0],
                  columns=["activation_vector", "transported_vectors",
                           "h42_recompute_cosine"])
rows = [r for r in t.slice(0, args.n_rows * 2).to_pylist()
        if r["h42_recompute_cosine"] >= 0.99][: args.n_rows]
H = torch.tensor(np.array([r["activation_vector"] for r in rows], dtype=np.float32),
                 device=dev)
L = torch.tensor(np.array([np.frombuffer(r["transported_vectors"], np.float16)
                           .reshape(16, -1)[:K] for r in rows], dtype=np.float32),
                 device=dev)
print(f"{H.shape[0]} rows", flush=True)

CANDS = {}
for hi in (8, 16):
    CANDS[f"uniform d<{hi}"] = [1.0] * hi + [0.0] * (K - hi)
    CANDS[f"uniform 1<=d<{hi}"] = [0.0] + [1.0] * (hi - 1) + [0.0] * (K - hi)
    CANDS[f"norm-equalised d<{hi}"] = [1.0 / nrm[d] for d in range(hi)] + [0.0] * (K - hi)
    CANDS[f"norm-eq 1<=d<{hi}"] = ([0.0] + [1.0 / nrm[d] for d in range(1, hi)]
                                   + [0.0] * (K - hi))
CANDS["taper 0.7^d, d<8"] = [0.7 ** d for d in range(8)] + [0.0] * (K - 8)
CANDS["d=0 only (the J-lens vector)"] = [1.0] + [0.0] * (K - 1)

out = []
for name, w in CANDS.items():
    W = torch.tensor(w, device=dev)
    JP = sum(W[d] * Jb[d] for d in range(K) if w[d])
    v_avg = (JP @ H.T).T
    v_loc = (L * W.view(1, -1, 1)).sum(1)
    mu_a, mu_l = v_avg.mean(0), v_loc.mean(0)
    a = float(cos(v_avg, v_loc, dim=-1).mean())
    h = float(cos(H, v_loc, dim=-1).mean())
    out.append({
        "weights": name,
        "cos_avg_local": a,
        "cos_h42_local": h,
        "margin": a - h,
        "cos_centered": float(cos(v_avg - mu_a, v_loc - mu_l, dim=-1).mean()),
        "cos_avg_to_mean": float(cos(v_avg, mu_a.expand_as(v_avg), dim=-1).mean()),
        "w": w,
    })

out.sort(key=lambda r: -r["margin"])
print(f"\n{'weights':30s} cos(avg,local)  cos(h42,local)   margin  centered  |to-mean")
for r in out:
    flag = "  <-- Jacobian earns its keep" if r["margin"] > 0.02 else ""
    print(f"{r['weights']:30s}    {r['cos_avg_local']:+.3f}         "
          f"{r['cos_h42_local']:+.3f}       {r['margin']:+.3f}   "
          f"{r['cos_centered']:+.3f}    {r['cos_avg_to_mean']:+.3f}{flag}")

best = out[0]
print(f"\nBEST: {best['weights']}  margin {best['margin']:+.3f}")
print("A positive margin means the averaged Jacobian carries something about the")
print("training target that raw h42 does not — the minimum for this lens to be a")
print("Jacobian lens rather than an elaborate way of feeding the activation.")
json.dump(out, open(args.out, "w"), indent=1)
print(f"wrote {args.out}", flush=True)
