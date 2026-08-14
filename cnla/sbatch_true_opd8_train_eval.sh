#!/bin/bash
#SBATCH --job-name=skiplens_true_opd8
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%A_%a.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
mkdir -p "$ROOT/results/true_opd8"

train_eval_lane() {
  local lane=$1
  local warm train val budget opd_ckpt sft_ckpt
  if [ "$lane" = normal ]; then
    warm=$ROOT/checkpoints/normal_warm/iter_0000200
    train=$ROOT/data/normal_train.parquet
    val=$ROOT/data/normal_val.parquet
    budget=10016
    opd_ckpt=$ROOT/checkpoints/normal_true_opd8
    sft_ckpt=$ROOT/checkpoints/normal_sft8_tokens/iter_0000626
  elif [ "$lane" = repeat ]; then
    # The original 200-step repeat warm start emitted EOS on 99.2% of held-out
    # activations. This 10,004-token SFT checkpoint is the first common policy
    # that passed the 0%-EOS support gate.
    warm=$ROOT/checkpoints/repeat_sft8_tokens/iter_0000917
    train=$ROOT/data/repeat_train.parquet
    val=$ROOT/data/repeat_val.parquet
    budget=10004
    opd_ckpt=$ROOT/checkpoints/repeat_true_opd8_strongwarm
    sft_ckpt=$ROOT/checkpoints/repeat_sft8_strongwarm/iter_0000917
  else
    echo "unknown lane: $lane" >&2
    return 2
  fi

  python -m nla.train_opd --objective opd --base-ckpt "$BASE" \
    --av-ckpt "$warm" --parquet "$train" --sidecar "$train" \
    --save-dir "$opd_ckpt" \
    --num-steps 20000 --lr-decay-steps 20000 --batch-size 2 \
    --max-new-tokens 8 --max-optimized-tokens "$budget" \
    --temperature 1.0 --lr 3e-5 --save-every 500 \
    --wandb-project skip-lens-opd --wandb-group true-opd8-vs-sft \
    --wandb-name "${lane}_true_opd8"

  local final_opd
  final_opd=$(find "$opd_ckpt" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' | sort | tail -n 1)
  test -n "$final_opd"
  local checkpoint arm feed label
  for arm in opd sft; do
    if [ "$arm" = opd ]; then checkpoint=$final_opd; else checkpoint=$sft_ckpt; fi
    for feed in activation_vector act_L42; do
      if [ "$feed" = activation_vector ]; then label=L62; else label=L42; fi
      python -m evals.opd_eval --base-ckpt "$BASE" --av-ckpt "$checkpoint" \
        --parquet "$val" --sidecar "$val" --feed-col "$feed" \
        --max-rows 256 --max-new-tokens 8 --batch-size 4 \
        --out "$ROOT/results/true_opd8/${lane}_${arm}_${label}_eval.json"
    done
  done
}

lanes=(normal repeat)
train_eval_lane "${lanes[$SLURM_ARRAY_TASK_ID]}"
echo "TRUE_OPD8_TRAIN_EVAL_DONE lane=${lanes[$SLURM_ARRAY_TASK_ID]}"
