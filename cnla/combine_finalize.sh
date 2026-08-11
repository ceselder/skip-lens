#!/bin/bash
# After the 6 datagen shards finish: concat them -> one collect parquet, then finalize into
# (activation, single-continuation) naive future-lens SFT pairs. Run on B300.
set -uo pipefail
cd /workspace/cnla/skip-lens && source /workspace/cnla_venv/bin/activate
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
python - <<'PY'
import pyarrow.parquet as pq, pyarrow as pa, json, glob
shards = sorted(glob.glob("data/fl_big/shard_*.parquet"))
allt = pa.concat_tables([pq.read_table(s) for s in shards])
pq.write_table(allt, "data/fl_big/collect_all.parquet", row_group_size=2000)
m = json.load(open(shards[0] + ".meta.json")); m["rows"] = allt.num_rows
json.dump(m, open("data/fl_big/collect_all.parquet.meta.json", "w"))
print(f"combined {allt.num_rows} rows from {len(shards)} shards")
PY
python pretrain/finalize_naive_data.py \
  --labeled data/fl_big/collect_all.parquet --meta data/fl_big/collect_all.parquet.meta.json \
  --out-train data/fl_big/sft_train.parquet --out-val data/fl_big/sft_val.parquet --rollout-idx 0
python -c "import pyarrow.parquet as pq; print('SFT_TRAIN_ROWS', pq.ParquetFile('data/fl_big/sft_train.parquet').metadata.num_rows)"
echo COMBINE_FINALIZE_DONE
