"""Per-space mean and whitening matrices for source-side standardization.

The slot families transport an activation from a SOURCE layer. If we want the
train-side and test-side slot distributions to be comparable, the right thing to
standardize is the source activation IN ITS OWN SPACE, on both sides:

    x_src = Sigma_src^{-1/2} (h_src - mu_src)        then  slots[d] = Jbar^(d) x_src

Both families then receive identity-covariance, zero-mean inputs, so the only
remaining difference between train and test is the transport operator itself
(which is the skip-lens variable we actually want to study).

Earlier runs centered on ONE side only (test-time L42) and never whitened,
which made the "centered" condition a transformation the decoder had never seen
— so its poor score (0.218) did not test the hypothesis it was meant to test.

Writes, per source layer L:
    hbar_L{L}.npy         mean activation                    [d]
    whiten_L{L}.npy       Sigma^{-1/2} (ridge-regularised)   [d, d]
    unwhiten_L{L}.npy     Sigma^{+1/2}                       [d, d]
    standardizer_diagnostics.json

Reads act_L{L} straight from the pass-1 shards, in fp64, one pass.
"""
import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-glob", default="/workspace/data/spans_raw/shard_[0-3].parquet")
    ap.add_argument("--layers", default="42,62")
    ap.add_argument("--out-dir", default="/workspace/results/offset_jlens")
    ap.add_argument("--ridge", type=float, default=1e-3,
                    help="ridge as a FRACTION of mean eigenvalue (Sigma + lam*I)")
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(args.raw_glob))
    assert files, f"no shards matched {args.raw_glob}"
    os.makedirs(args.out_dir, exist_ok=True)

    diag = {"n_rows": {}, "layers": {}, "ridge_frac": args.ridge}
    for L in layers:
        col = f"act_L{L}"
        d = None
        s1 = None          # sum of h            [d]
        s2 = None          # sum of h h^T        [d, d]
        n = 0
        for f in files:
            pf = pq.ParquetFile(f)
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg, columns=[col])
                A = np.asarray(t.column(col).to_pylist(), dtype=np.float32)
                for b in range(0, len(A), args.batch):
                    X = torch.from_numpy(A[b:b + args.batch]).to(dev).double()
                    if d is None:
                        d = X.shape[1]
                        s1 = torch.zeros(d, dtype=torch.float64, device=dev)
                        s2 = torch.zeros(d, d, dtype=torch.float64, device=dev)
                    s1 += X.sum(0)
                    s2 += X.T @ X
                    n += X.shape[0]
            print(f"  L{L}: {n} rows after {os.path.basename(f)}", flush=True)

        mu = s1 / n
        cov = s2 / n - torch.outer(mu, mu)
        cov = 0.5 * (cov + cov.T)                       # symmetrise fp noise
        evals, evecs = torch.linalg.eigh(cov)
        lam = args.ridge * float(evals.mean())
        ev = torch.clamp(evals, min=0.0) + lam
        W = (evecs * ev.rsqrt()) @ evecs.T              # Sigma^{-1/2}
        U = (evecs * ev.sqrt()) @ evecs.T               # Sigma^{+1/2}

        np.save(f"{args.out_dir}/hbar_L{L}.npy", mu.float().cpu().numpy())
        np.save(f"{args.out_dir}/whiten_L{L}.npy", W.float().cpu().numpy())
        np.save(f"{args.out_dir}/unwhiten_L{L}.npy", U.float().cpu().numpy())

        # sanity: whitened data should have ~unit variance per dim
        checkX = torch.from_numpy(A[:4096]).to(dev).double()
        z = (checkX - mu) @ W.T
        entry = {"n_rows": n, "d_model": d, "mean_norm": float(mu.norm()),
                 "eig_min": float(evals.min()), "eig_max": float(evals.max()),
                 "eig_mean": float(evals.mean()), "ridge_abs": lam,
                 "cond_after_ridge": float((ev.max() / ev.min()).item()),
                 "whitened_var_mean": float(z.var(dim=0).mean()),
                 "whitened_var_std": float(z.var(dim=0).std())}
        diag["layers"][str(L)] = entry
        diag["n_rows"][str(L)] = n
        print(f"L{L}: ||mu||={entry['mean_norm']:.1f}  eig {entry['eig_min']:.3g}"
              f"..{entry['eig_max']:.3g} (mean {entry['eig_mean']:.3g})  "
              f"cond={entry['cond_after_ridge']:.1f}  "
              f"whitened var {entry['whitened_var_mean']:.3f}"
              f"±{entry['whitened_var_std']:.3f}", flush=True)

    json.dump(diag, open(f"{args.out_dir}/standardizer_diagnostics.json", "w"), indent=2)
    print(f"wrote means + whiteners for layers {layers} to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
