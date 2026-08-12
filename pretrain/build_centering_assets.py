#!/usr/bin/env python3
"""Save the corpus-mean h42 and per-offset mean transported directions, for
centered/deflated test-time slot construction."""
import glob
import numpy as np
import pyarrow.parquet as pq

OUT = "/workspace/results/offset_jlens"
H, N = None, 0
for f in sorted(glob.glob("/workspace/data/spans_jvp/shard_[0-3]_jvp.parquet")):
    t = pq.read_table(f, columns=["activation_vector"])
    A = np.array(t.column("activation_vector").to_pylist(), dtype=np.float64)
    H = A.sum(0) if H is None else H + A.sum(0)
    N += A.shape[0]
    print(f"{f}: {A.shape[0]} rows (total {N})", flush=True)
hbar = (H / N).astype("float32")
np.save(f"{OUT}/hbar_L42.npy", hbar)
print("saved hbar_L42.npy  ||hbar|| =", float(np.linalg.norm(hbar)),
      " mean-cos of hbar with rows is what drove the shared component", flush=True)

# mean transported direction per offset, computed on the same corpus mean basis
for d in range(16):
    J = np.load(f"{OUT}/Jbar_L42_to_L62_off{d}.npy")
    md = J @ hbar
    md = md / (np.linalg.norm(md) + 1e-9)
    np.save(f"{OUT}/meandir_off{d}.npy", md.astype("float32"))
print("saved meandir_off{0..15}.npy", flush=True)
