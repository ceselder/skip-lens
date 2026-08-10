#!/bin/bash
#SBATCH --job-name=skiplens_repeat_collect
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
DATA=$ROOT/data/repeat_scaled_all_50k

source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
export DATA
mkdir -p "$DATA" "$ROOT/logs"
cd "$SRC"

# Four independent OS-CSPRNG streams, 12,500 phrases each. Phrase hashes define
# each shard's train/validation split; generated phrases are overwhelmingly
# unique, and the merged split is checked before training.
for shard in 0 1 2 3; do
  srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 \
    --cpus-per-task=8 --mem=120G \
    python -m pretrain.collect_repeat_data --base-model "$BASE" \
      --out-train "$DATA/shard_${shard}_train.parquet" \
      --out-val "$DATA/shard_${shard}_val.parquet" \
      --n-phrases 12500 --phrase-words 40 --positions-per-phrase 4 \
      --max-span 16 --layers 42 62 --target-layer 62 --batch-size 16 &
done
wait

python -m pretrain.merge_repeat_shards \
  --inputs "$DATA"/shard_*_train.parquet --out "$DATA/train.parquet"
python -m pretrain.merge_repeat_shards \
  --inputs "$DATA"/shard_*_val.parquet --out "$DATA/val.parquet"

python - <<'PY'
import os
import pyarrow.parquet as pq

root = os.environ.get(
    "DATA", "/workspace-vast/celeste/skip-lens-opd/data/repeat_scaled_all_50k"
)
train = set(pq.read_table(root + "/train.parquet", columns=["doc_id"])["doc_id"].to_pylist())
val = set(pq.read_table(root + "/val.parquet", columns=["doc_id"])["doc_id"].to_pylist())
overlap = train & val
if overlap:
    raise RuntimeError(f"phrase leakage across merged split: {len(overlap)} IDs")
print(f"merged phrase IDs: train={len(train)} val={len(val)} overlap=0")
PY

echo REPEAT_SCALED_COLLECTION_DONE
