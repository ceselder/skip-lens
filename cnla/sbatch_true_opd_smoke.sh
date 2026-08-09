#!/bin/bash
#SBATCH --job-name=skiplens_true_opd_smoke
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
cd "$SRC"

OUT=$ROOT/checkpoints/true_opd_smoke_${SLURM_JOB_ID}
python -m nla.train_opd \
  --objective opd --base-ckpt Qwen/Qwen3.6-27B \
  --av-ckpt "$ROOT/checkpoints/normal_warm/iter_0000200" \
  --parquet "$ROOT/data/normal_train.parquet" \
  --sidecar "$ROOT/data/normal_train.parquet.nla_meta.yaml" \
  --save-dir "$OUT" --num-steps 2 --batch-size 2 \
  --max-new-tokens 8 --max-rows 8 --save-every 1 \
  --temperature 1.0 --no-wandb

test -f "$OUT/iter_0000002/adapter_model.safetensors"
echo TRUE_OPD_SMOKE_DONE
