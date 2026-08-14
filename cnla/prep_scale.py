"""Scale test data: 5k-pair versions of the raw and normalized-mean-centering arms (same first-5k
fl_big rows, so the only variables are data size (500->5k) and the injected representation)."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SRC = "/workspace/cnla/skip-lens/data/fl_big/av_L62_150k.parquet"
OUT = "/workspace/cnla/skip-lens/data/meansub"
MEAN = f"{OUT}/mean_dir_L62_train.npy"
N = 5000

pf = pq.ParquetFile(SRC)
parts, have = [], 0
for rg in range(pf.num_row_groups):
    parts.append(pf.read_row_group(rg))
    have += parts[-1].num_rows
    if have >= N:
        break
full = pa.concat_tables(parts).slice(0, N)
side = yaml.safe_load(open(SRC + ".nla_meta.yaml"))
side["row_count"] = N

# raw arm (unchanged L62 activations)
pq.write_table(full, f"{OUT}/av_raw_5k.parquet")
yaml.safe_dump(side, open(f"{OUT}/av_raw_5k.parquet.nla_meta.yaml", "w"), sort_keys=False)

# normalized mean-centering arm: h/||h|| - mean_dir_L62_train
av = np.asarray(full["activation_vector"].to_pylist(), dtype=np.float32)
mean_dir = np.load(MEAN).astype(np.float32)
u = av / (np.linalg.norm(av, axis=1, keepdims=True) + 1e-8)
z = (u - mean_dir[None, :]).astype(np.float32)
idx = full.schema.get_field_index("activation_vector")
col = pa.array([a.tolist() for a in z], type=full.schema.field("activation_vector").type)
tb = full.set_column(idx, "activation_vector", col)
pq.write_table(tb, f"{OUT}/av_meansub_norm_5k.parquet")
yaml.safe_dump(side, open(f"{OUT}/av_meansub_norm_5k.parquet.nla_meta.yaml", "w"), sort_keys=False)
print(f"wrote av_raw_5k + av_meansub_norm_5k (N={N})")
