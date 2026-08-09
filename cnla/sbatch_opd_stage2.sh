#!/bin/bash
#SBATCH --job-name=skiplens_stage2
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=18:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
TOKEN_BUDGET=${TOKEN_BUDGET:-10000}
source /workspace-vast/celeste/.keys.env
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC

run_lane() {
  local name=$1
  local train=$ROOT/data/${name}_train.parquet
  local val=$ROOT/data/${name}_val.parquet
  local warm=$ROOT/checkpoints/${name}_warm/iter_0000200
  local result=$ROOT/results/${name}
  mkdir -p "$result"

  python -m evals.opd_eval --base-ckpt "$BASE" --av-ckpt "$warm" \
    --parquet "$val" --sidecar "$val" --max-rows 256 \
    --out "$result/warm_eval.json"
  python scripts/select_kl_threshold.py --eval "$result/warm_eval.json" \
    --out "$result/kl_threshold.json"
  local threshold
  threshold=$(python -c "import json; print(json.load(open('$result/kl_threshold.json'))['threshold_nats'])")

  python -m nla.train_opd --objective forward_kl --base-ckpt "$BASE" \
    --av-ckpt "$warm" --parquet "$train" --sidecar "$train" \
    --save-dir "$ROOT/checkpoints/${name}_opd" \
    --num-steps 20000 --lr-decay-steps 20000 --batch-size 2 --max-new-tokens 16 \
    --kl-threshold "$threshold" --max-optimized-tokens "$TOKEN_BUDGET" \
    --lr 3e-5 --save-every 500 \
    --wandb-project skip-lens-opd --wandb-group "${name}-opd-vs-sft" \
    --wandb-name "${name}_opd"

  local opd_tokens opd_seconds
  opd_tokens=$(python -c "import json; print(json.load(open('$ROOT/checkpoints/${name}_opd/run_summary.json'))['optimized_tokens'])")
  opd_seconds=$(python -c "import json; print(json.load(open('$ROOT/checkpoints/${name}_opd/run_summary.json'))['wall_seconds'])")

  python -m nla.train_opd --objective sft --base-ckpt "$BASE" \
    --av-ckpt "$warm" --parquet "$train" --sidecar "$train" \
    --save-dir "$ROOT/checkpoints/${name}_sft_tokens" \
    --num-steps 20000 --lr-decay-steps 20000 --batch-size 2 --max-new-tokens 16 \
    --max-optimized-tokens "$opd_tokens" --lr 3e-5 --save-every 1000 \
    --wandb-project skip-lens-opd --wandb-group "${name}-opd-vs-sft" \
    --wandb-name "${name}_sft_token_matched"

  python -m nla.train_opd --objective sft --base-ckpt "$BASE" \
    --av-ckpt "$warm" --parquet "$train" --sidecar "$train" \
    --save-dir "$ROOT/checkpoints/${name}_sft_time" \
    --num-steps 100000 --lr-decay-steps 20000 --batch-size 4 --max-new-tokens 16 \
    --max-wall-seconds "$opd_seconds" --lr 3e-5 --save-every 1000 \
    --wandb-project skip-lens-opd --wandb-group "${name}-opd-vs-sft" \
    --wandb-name "${name}_sft_time_matched"
}

export -f run_lane
export ROOT SRC VENV BASE TOKEN_BUDGET HF_HOME HF_TOKEN_PATH PYTHONPATH

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; run_lane normal" &
normal_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; run_lane repeat" &
repeat_pid=$!
wait "$normal_pid"
wait "$repeat_pid"
echo STAGE2_DONE
