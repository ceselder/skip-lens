"""Build the mean-subtraction ablation data.

Two 500-pair AV-SFT parquets sliced from fl_big, IDENTICAL except for the activation representation:
  av_raw_500      : activation_vector = raw L62 residual        (control)
  av_meansub_500  : activation_vector = raw L62 - mean_L62      (mean-centered: inject the DEVIATION)
mean_L62 is estimated over the first N_MEAN fl_big rows (the training distribution). Also writes
mean_L62_train.npy and copies the injection sidecar (row_count -> 500) for both parquets so
train_sft finds the injection char/token unchanged.
"""
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

SRC = "/workspace/cnla/skip-lens/data/fl_big/av_L62_150k.parquet"
OUT = "/workspace/cnla/skip-lens/data/meansub"
N_MEAN = 20000     # rows to estimate the training-distribution mean over
N_TRAIN = 500      # span-activation pairs to train on (as requested)
os.makedirs(OUT, exist_ok=True)

pf = pq.ParquetFile(SRC)
parts, have = [], 0
for rg in range(pf.num_row_groups):
    parts.append(pf.read_row_group(rg))
    have += parts[-1].num_rows
    if have >= N_MEAN:
        break
big = pa.concat_tables(parts)

av_all = np.asarray(big["activation_vector"].to_pylist()[:N_MEAN], dtype=np.float32)  # (N_MEAN,5120)
mean = av_all.mean(0).astype(np.float32)
np.save(f"{OUT}/mean_L62_train.npy", mean)
print(f"mean_L62 over {av_all.shape[0]} rows | ||mean||={np.linalg.norm(mean):.2f} "
      f"| mean per-row ||act||={np.linalg.norm(av_all, axis=1).mean():.2f}")

full = big.slice(0, N_TRAIN)
av_raw = np.asarray(full["activation_vector"].to_pylist(), dtype=np.float32)  # (500,5120)
av_sub = (av_raw - mean[None, :]).astype(np.float32)
print(f"train pairs={av_raw.shape[0]} | mean-sub per-row ||dev||={np.linalg.norm(av_sub, axis=1).mean():.2f} "
      f"(raw ||act||={np.linalg.norm(av_raw, axis=1).mean():.2f})")


def replace_av(tb, av):
    idx = tb.schema.get_field_index("activation_vector")
    col = pa.array([a.tolist() for a in av], type=tb.schema.field("activation_vector").type)
    return tb.set_column(idx, "activation_vector", col)


side = yaml.safe_load(open(SRC + ".nla_meta.yaml"))
side["row_count"] = N_TRAIN
for name, av in [("av_raw_500", av_raw), ("av_meansub_500", av_sub)]:
    path = f"{OUT}/{name}.parquet"
    pq.write_table(replace_av(full, av), path)
    yaml.safe_dump(side, open(path + ".nla_meta.yaml", "w"), sort_keys=False)
    print("wrote", path)
