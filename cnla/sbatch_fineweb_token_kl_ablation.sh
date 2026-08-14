#!/bin/bash
#SBATCH --job-name=skiplens_token_kl
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
DATA=$ROOT/data/fineweb_t1_half
WARM=$ROOT/checkpoints/fineweb_t1_disjoint/warm_sft_10k/iter_0000625
CKPTS=$ROOT/checkpoints/fineweb_token_kl
RESULTS=$ROOT/results/fineweb_token_kl
TOKEN_BUDGET=10000

source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
mkdir -p "$CKPTS" "$RESULTS" "$ROOT/logs"
cd "$SRC"

train_eval() {
  local name=$1
  local topk=$2
  python -m nla.train_opd --objective token_kl --teacher-top-k "$topk" \
    --base-ckpt "$BASE" --av-ckpt "$WARM" \
    --parquet "$DATA/stage2_train.parquet" --sidecar "$DATA/stage2_train.parquet" \
    --save-dir "$CKPTS/$name" --num-steps 20000 --lr-decay-steps 20000 \
    --batch-size 2 --max-new-tokens 8 --max-optimized-tokens "$TOKEN_BUDGET" \
    --lr 3e-5 --min-lr 3e-5 --save-every 500 \
    --wandb-project skip-lens-opd --wandb-group fineweb-token-kl \
    --wandb-name "fineweb_${name}"
  local checkpoint feed label
  checkpoint=$(find "$CKPTS/$name" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' | sort | tail -n 1)
  test -n "$checkpoint"
  for feed in activation_vector act_L42; do
    if [ "$feed" = activation_vector ]; then label=L62; else label=L42; fi
    python -m evals.opd_eval --base-ckpt "$BASE" --av-ckpt "$checkpoint" \
      --parquet "$DATA/val.parquet" --sidecar "$DATA/val.parquet" \
      --feed-col "$feed" --max-rows 256 --max-new-tokens 8 --batch-size 4 \
      --out "$RESULTS/fineweb_${name}_${label}_eval.json"
  done
}
export -f train_eval
export ROOT SRC VENV BASE DATA WARM CKPTS RESULTS TOKEN_BUDGET HF_HOME HF_TOKEN_PATH PYTHONPATH

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc 'source /workspace-vast/celeste/.keys.env; source "$VENV/bin/activate"; cd "$SRC"; train_eval token_kl_exact 0' &
exact_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc 'source /workspace-vast/celeste/.keys.env; source "$VENV/bin/activate"; cd "$SRC"; train_eval token_kl_top32 32' &
topk_pid=$!
wait "$exact_pid"
wait "$topk_pid"
echo FINEWEB_TOKEN_KL_ABLATION_DONE
