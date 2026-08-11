#!/usr/bin/env python3
"""Slice the first N rows of fl_big into a token-matched AV-SFT subset (activation = L62, as-is).
This is the L62 arm of the source-layer ablation; its L42 twin is built by extract_L42_subset.py."""
import sys, os, shutil
import pyarrow as pa, pyarrow.parquet as pq

SRC = "/workspace/cnla/skip-lens/data/fl_big/sft_train.parquet"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
OUT = f"/workspace/cnla/skip-lens/data/fl_big/av_L62_{N // 1000}k.parquet"

pf = pq.ParquetFile(SRC)
batches, got = [], 0
for b in pf.iter_batches(batch_size=2000):
    batches.append(b)
    got += b.num_rows
    if got >= N:
        break
t = pa.Table.from_batches(batches).slice(0, N)
pq.write_table(t, OUT, row_group_size=2000)
sc = SRC + ".nla_meta.yaml"
if os.path.exists(sc):
    shutil.copy(sc, OUT + ".nla_meta.yaml")
print(f"wrote {OUT} ({t.num_rows} rows), sidecar={'copied' if os.path.exists(sc) else 'MISSING'}", flush=True)
