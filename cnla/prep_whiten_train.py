"""Whitening ablation, TRAIN side. ZCA-whiten the L62 training distribution:
z = Sigma^{-1/2} (h - mu), with mu, Sigma from the stored fl_big L62 activations and shrinkage on
Sigma (alpha=0.15) so Sigma^{-1/2} does not amplify noise directions. Store the whitened 500-pair
training vectors (injection is norm-matched, so only direction matters). Saves whiten_train_L62.npz.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SRC = "/workspace/cnla/skip-lens/data/fl_big/av_L62_150k.parquet"
OUT = "/workspace/cnla/skip-lens/data/meansub"
N_STATS = 20000
N_TRAIN = 500
RIDGE = 0.1   # lambda as a fraction of the mean eigenvalue: W = (Sigma + lambda*I)^(-1/2)

pf = pq.ParquetFile(SRC)
parts, have = [], 0
for rg in range(pf.num_row_groups):
    parts.append(pf.read_row_group(rg))
    have += parts[-1].num_rows
    if have >= N_STATS:
        break
big = pa.concat_tables(parts)

av = np.asarray(big["activation_vector"].to_pylist()[:N_STATS], dtype=np.float32)
mu = av.mean(0)
Xc = av - mu
S = (Xc.T @ Xc) / len(Xc)
S = (1 - ALPHA) * S + ALPHA * (np.trace(S) / S.shape[0]) * np.eye(S.shape[0], dtype=S.dtype)
w, U = np.linalg.eigh(S.astype(np.float64))
W = ((U * (w ** -0.5)) @ U.T).astype(np.float32)          # ZCA whitening matrix (symmetric)
np.savez(f"{OUT}/whiten_train_L62.npz", mu=mu.astype(np.float32), W=W)
print(f"L62 ZCA: cond={w.max()/w.min():.1f} ||mu||={np.linalg.norm(mu):.1f} eig[min,max]=[{w.min():.3g},{w.max():.3g}]")

full = big.slice(0, N_TRAIN)
h = np.asarray(full["activation_vector"].to_pylist(), dtype=np.float32)
z = ((h - mu) @ W).astype(np.float32)                     # W symmetric so (h-mu)@W == W@(h-mu)
print(f"whitened train pairs={z.shape[0]} | mean ||z||={np.linalg.norm(z, axis=1).mean():.3f}")

idx = full.schema.get_field_index("activation_vector")
col = pa.array([a.tolist() for a in z], type=full.schema.field("activation_vector").type)
tb = full.set_column(idx, "activation_vector", col)
path = f"{OUT}/av_whiten_500.parquet"
pq.write_table(tb, path)
side = yaml.safe_load(open(SRC + ".nla_meta.yaml"))
side["row_count"] = N_TRAIN
yaml.safe_dump(side, open(path + ".nla_meta.yaml", "w"), sort_keys=False)
print("wrote", path)
