"""Does J̄⁽ᐞ⁾h actually DEPEND on h? (the "always says aircraft" test)

Symptom: the multi-slot lens emits near-identical text regardless of which
position it reads. That happens if the transported direction is essentially
context-independent — i.e. the averaged Jacobian is dominated by one mean
direction, so J̄h ≈ c(h) * u1 and the per-slot norm-matched injection erases
c(h), handing the decoder the same vector every time.

Measures, over many real h42:
  across-h cosine of normalized J̄⁽ᐞ⁾h   (LOW = informative, HIGH = blind)
  the same for the stored LOCAL transports (the training distribution)
  the same for raw h42 itself (control)
and for two candidate fixes:
  centered   : J̄⁽ᐞ⁾(h - h̄)             (h̄ = corpus mean h42)
  deflated   : J̄⁽ᐞ⁾h with the mean transported direction projected out
Also reports the spectral concentration of J̄⁽ᐞ⁾ (top singular value share)
and how much of J̄⁽ᐞ⁾h's norm lies along the mean transported direction.
"""
import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--jvp", default="/workspace/data/spans_jvp/shard_0_jvp.parquet")
ap.add_argument("--n-rows", type=int, default=512)
ap.add_argument("--deltas", default="0,1,3,7")
ap.add_argument("--out", default="/workspace/results/multislot_eval/across_h_variance.json")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
DELTAS = [int(x) for x in args.deltas.split(",")]

t = pq.read_table(args.jvp, columns=["activation_vector", "transported_vectors"])
rows = t.slice(0, args.n_rows).to_pylist()
H = torch.tensor(np.array([r["activation_vector"] for r in rows], dtype=np.float32),
                 device=dev)                                   # [N, d]
LOC = torch.tensor(np.array([
    np.frombuffer(r["transported_vectors"], dtype=np.float16).reshape(16, -1)
    for r in rows], dtype=np.float32), device=dev)              # [N, 16, d]
N, d = H.shape
print(f"{N} rows, d={d}", flush=True)
Hbar = H.mean(0, keepdim=True)


def across_h_cos(X):
    """X: [N, d] -> mean pairwise cosine between rows (sampled)."""
    U = torch.nn.functional.normalize(X, dim=-1)
    C = U @ U.T
    iu = torch.triu_indices(N, N, offset=1, device=X.device)
    return float(C[iu[0], iu[1]].mean()), float(C[iu[0], iu[1]].std())


res = {"n_rows": N, "per_delta": {}}
m, s = across_h_cos(H)
res["raw_h42_across_h_cos"] = [m, s]
print(f"raw h42            across-h cos = {m:+.3f} ± {s:.3f}", flush=True)

for dl in DELTAS:
    J = torch.from_numpy(np.load(
        f"{args.jbar_dir}/Jbar_L42_to_L62_off{dl}.npy")).float().to(dev)
    T = H @ J.T                                    # [N, d] averaged transports
    Tc = (H - Hbar) @ J.T                          # centered
    mean_dir = torch.nn.functional.normalize(T.mean(0), dim=-1)
    along = (T @ mean_dir).abs() / T.norm(dim=-1)  # share of norm along mean dir
    Td = T - (T @ mean_dir)[:, None] * mean_dir[None, :]   # deflated
    L = LOC[:, dl, :]                              # local transports, same delta

    sv = torch.linalg.svdvals(J)
    top1 = float((sv[0] ** 2) / (sv ** 2).sum())

    entry = {}
    for name, X in (("averaged", T), ("centered", Tc), ("deflated", Td),
                    ("local", L)):
        m, s = across_h_cos(X)
        entry[name] = [m, s]
    entry["mean_dir_norm_share"] = [float(along.mean()), float(along.std())]
    entry["J_top1_var_share"] = top1
    res["per_delta"][dl] = entry
    print(f"\ndelta={dl}  (J top-1 singular var share {top1:.3f}, "
          f"|proj on mean dir| = {float(along.mean()):.3f})", flush=True)
    for name in ("averaged", "centered", "deflated", "local"):
        print(f"   {name:9s} across-h cos = {entry[name][0]:+.3f} ± {entry[name][1]:.3f}",
              flush=True)

json.dump(res, open(args.out, "w"), indent=2)
print(f"\nwrote {args.out}", flush=True)
worst = res["per_delta"][DELTAS[0]]["averaged"][0]
print("VERDICT:", "BLIND — averaged transports barely depend on h; the decoder "
      "sees ~one fixed vector (explains constant readouts)"
      if worst > 0.9 else
      ("PARTIALLY BLIND — strong shared component" if worst > 0.6
       else "informative — h-dependence is healthy"), flush=True)
