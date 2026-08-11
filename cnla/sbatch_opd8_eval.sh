#!/bin/bash
#SBATCH --job-name=skiplens_opd8_eval
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
source /workspace-vast/celeste/.keys.env
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
mkdir -p "$ROOT/results/opd8"

latest_checkpoint() {
  find "$1" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' | sort | tail -n 1
}

eval_lane() {
  local dataset=$1
  local parquet=$ROOT/data/${dataset}_val.parquet
  local warm=$ROOT/checkpoints/${dataset}_warm/iter_0000200
  local arm checkpoint feed label
  for arm in warm opd8 sft8_tokens sft8_time; do
    if [ "$arm" = warm ]; then
      checkpoint=$warm
    else
      checkpoint=$(latest_checkpoint "$ROOT/checkpoints/${dataset}_${arm}")
    fi
    test -n "$checkpoint"
    for feed in activation_vector act_L42; do
      if [ "$feed" = activation_vector ]; then label=L62; else label=L42; fi
      python -m evals.opd_eval --base-ckpt "$BASE" --av-ckpt "$checkpoint" \
        --parquet "$parquet" --sidecar "$parquet" --feed-col "$feed" \
        --max-rows 256 --max-new-tokens 8 --batch-size 4 \
        --out "$ROOT/results/opd8/${dataset}_${arm}_${label}_eval.json"
    done
  done
}

export -f latest_checkpoint eval_lane
export ROOT SRC VENV BASE HF_HOME HF_TOKEN_PATH PYTHONPATH
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; eval_lane normal" &
normal_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; eval_lane repeat" &
repeat_pid=$!
wait "$normal_pid"
wait "$repeat_pid"

source "$VENV/bin/activate"
python scripts/merge_eval_json.py \
  --inputs "$ROOT"/results/opd8/normal_*_eval.json \
  --out "$ROOT/results/opd8/normal_quality_inputs.json"
python -m evals.judge_opd_quality \
  --input "$ROOT/results/opd8/normal_quality_inputs.json" \
  --out "$ROOT/results/opd8/normal_quality_judged.json"
echo OPD8_EVAL_DONE
