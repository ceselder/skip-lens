#!/bin/bash
#SBATCH --job-name=skiplens_opd_scale_data
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%A_%a.out
set -euo pipefail

# Eight non-overlapping shards: 20,000 FineWeb documents × five uniformly
# selected positions/document = up to 100,000 activation/teacher-context rows.

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
DATA=$ROOT/data/fineweb_t1_scaled_100k
DOCS_PER_SHARD=2500
OFFSET=$((SLURM_ARRAY_TASK_ID * DOCS_PER_SHARD))
OUT=$DATA/shards/collected_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").parquet

source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
mkdir -p "$DATA/shards" "$ROOT/logs"
cd "$SRC"

python -m pretrain.collect_ao_data \
  --base-ckpt "$BASE" \
  --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
  --layers 62 42 --n-docs "$DOCS_PER_SHARD" --doc-offset "$OFFSET" \
  --positions-per-doc 5 --rollouts 1 --rollout-len 8 \
  --temperature 1.0 --top-p 1.0 --no-decision-points \
  --seed $((20260813 + SLURM_ARRAY_TASK_ID)) --out "$OUT"

echo "SCALED_FINEWEB_SHARD_DONE shard=$SLURM_ARRAY_TASK_ID out=$OUT"
