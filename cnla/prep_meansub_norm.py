"""Mean-subtraction ablation done in NORMALIZED space (the fix): normalize each activation to unit
norm, take the mean THERE (mean direction), and subtract that — so the subtracted offset is not
dominated by magnitude (raw layer norms span 59->225, which made the raw-mean version cursed).

Training data: activation_vector = h/||h|| - mean_dir_L62_train, where mean_dir_L62_train is the mean
of unit L62 activations over the fl_big training distribution. (Injection norm-matches anyway, so only
the DIRECTION of the stored vector matters — the raw control already = 'no subtraction'.)
"""
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SRC = "/workspace/cnla/skip-lens/data/fl_big/av_L62_150k.parquet"
OUT = "/workspace/cnla/skip-lens/data/meansub"
N_MEAN = 20000
N_TRAIN = 500
os.makedirs(OUT, exist_ok=True)

pf = pq.ParquetFile(SRC)
parts, have = [], 0
for rg in range(pf.num_row_groups):
    parts.append(pf.read_row_group(rg))
    have += parts[-1].num_rows
    if have >= N_MEAN:
        break
big = pa.concat_tables(parts)

av = np.asarray(big["activation_vector"].to_pylist()[:N_MEAN], dtype=np.float32)
U = av / (np.linalg.norm(av, axis=1, keepdims=True) + 1e-8)
mean_dir = U.mean(0).astype(np.float32)
np.save(f"{OUT}/mean_dir_L62_train.npy", mean_dir)
print(f"mean_dir_L62_train: ||concentration||={np.linalg.norm(mean_dir):.4f} over {U.shape[0]} unit vecs")

full = big.slice(0, N_TRAIN)
av500 = np.asarray(full["activation_vector"].to_pylist(), dtype=np.float32)
u500 = av500 / (np.linalg.norm(av500, axis=1, keepdims=True) + 1e-8)
sub = (u500 - mean_dir[None, :]).astype(np.float32)
print(f"train pairs={sub.shape[0]} | per-row ||u - mean_dir||={np.linalg.norm(sub, axis=1).mean():.4f}")

idx = full.schema.get_field_index("activation_vector")
col = pa.array([a.tolist() for a in sub], type=full.schema.field("activation_vector").type)
tb = full.set_column(idx, "activation_vector", col)
path = f"{OUT}/av_meansub_norm_500.parquet"
pq.write_table(tb, path)
side = yaml.safe_load(open(SRC + ".nla_meta.yaml"))
side["row_count"] = N_TRAIN
yaml.safe_dump(side, open(path + ".nla_meta.yaml", "w"), sort_keys=False)
print("wrote", path)
