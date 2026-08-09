#!/bin/bash
#SBATCH --job-name=repeat_opd8_strongwarm
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
source "$VENV/bin/activate"
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
mkdir -p "$ROOT/results/opd8_strong_repeat"

latest_checkpoint() {
  find "$1" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' | sort | tail -n 1
}

WARM=$(latest_checkpoint "$ROOT/checkpoints/repeat_sft8_tokens")
test -n "$WARM"
TRAIN=$ROOT/data/repeat_train.parquet
VAL=$ROOT/data/repeat_val.parquet

train_arm() {
  local objective=$1
  local output=$2
  python -m nla.train_opd --objective "$objective" --base-ckpt "$BASE" \
    --av-ckpt "$WARM" --parquet "$TRAIN" --sidecar "$TRAIN" \
    --save-dir "$ROOT/checkpoints/$output" \
    --num-steps 20000 --lr-decay-steps 20000 --batch-size 2 \
    --max-new-tokens 8 --max-optimized-tokens "$TOKEN_BUDGET" \
    --lr 3e-5 --save-every 500 \
    --wandb-project skip-lens-opd --wandb-group repeat-opd8-strongwarm \
    --wandb-name "$output"
}

export -f train_arm latest_checkpoint
export ROOT SRC VENV BASE TOKEN_BUDGET WARM TRAIN VAL HF_HOME HF_TOKEN_PATH PYTHONPATH
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; train_arm forward_kl repeat_opd8_strongwarm" &
opd_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; train_arm sft repeat_sft8_strongwarm" &
sft_pid=$!
wait "$opd_pid"
wait "$sft_pid"

eval_arm() {
  local arm=$1
  local checkpoint
  if [ "$arm" = warm ]; then
    checkpoint=$WARM
  else
    checkpoint=$(latest_checkpoint "$ROOT/checkpoints/repeat_${arm}_strongwarm")
  fi
  local feed label
  for feed in activation_vector act_L42; do
    if [ "$feed" = activation_vector ]; then label=L62; else label=L42; fi
    python -m evals.opd_eval --base-ckpt "$BASE" --av-ckpt "$checkpoint" \
      --parquet "$VAL" --sidecar "$VAL" --feed-col "$feed" \
      --max-rows 256 --max-new-tokens 8 --batch-size 4 \
      --out "$ROOT/results/opd8_strong_repeat/repeat_${arm}_${label}_eval.json"
  done
}

export -f eval_arm
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; eval_arm opd8" &
opd_eval_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; eval_arm sft8" &
sft_eval_pid=$!
wait "$opd_eval_pid"
wait "$sft_eval_pid"

# Evaluate the common strong warm checkpoint once after both trained arms.
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; eval_arm warm"
echo REPEAT_OPD8_STRONGWARM_DONE
