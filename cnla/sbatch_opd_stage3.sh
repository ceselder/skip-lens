#!/bin/bash
#SBATCH --job-name=skiplens_stage3
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
JDIR=${JDIR:-/workspace-vast/celeste/multi-token-jlens-nla-lastlayer/results/jlens_official}
source /workspace-vast/celeste/.keys.env
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
mkdir -p "$ROOT/results/stage3"

latest_checkpoint() {
  find "$1" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' | sort | tail -n 1
}

eval_lane() {
  local dataset=$1
  local parquet=$ROOT/data/${dataset}_val.parquet
  local result=$ROOT/results/stage3
  local warm=$ROOT/checkpoints/${dataset}_warm/iter_0000200
  local opd sft_tokens sft_time
  opd=$(latest_checkpoint "$ROOT/checkpoints/${dataset}_opd")
  sft_tokens=$(latest_checkpoint "$ROOT/checkpoints/${dataset}_sft_tokens")
  sft_time=$(latest_checkpoint "$ROOT/checkpoints/${dataset}_sft_time")
  test -n "$opd" && test -n "$sft_tokens" && test -n "$sft_time"

  local arm checkpoint feed label
  for arm in warm opd sft_tokens sft_time; do
    case "$arm" in
      warm) checkpoint=$warm ;;
      opd) checkpoint=$opd ;;
      sft_tokens) checkpoint=$sft_tokens ;;
      sft_time) checkpoint=$sft_time ;;
    esac
    for feed in activation_vector act_L42; do
      if [ "$feed" = activation_vector ]; then label=L62; else label=L42; fi
      python -m evals.opd_eval --base-ckpt "$BASE" --av-ckpt "$checkpoint" \
        --parquet "$parquet" --sidecar "$parquet" --feed-col "$feed" \
        --max-rows 256 --max-new-tokens 16 --batch-size 4 \
        --out "$result/${dataset}_${arm}_${label}_eval.json"
    done
  done
}

export -f latest_checkpoint eval_lane
export ROOT SRC VENV BASE JDIR HF_HOME HF_TOKEN_PATH PYTHONPATH

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; eval_lane normal" &
normal_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; eval_lane repeat" &
repeat_pid=$!
wait "$normal_pid"
wait "$repeat_pid"

# The paper-suite comparison is the primary token-matched test. Both lanes use
# the exact six released prompt sets, paper positions, five layers, raw and
# Jacobian-transported vectors, plus same-layer shuffled controls.
opd_ckpt=$(latest_checkpoint "$ROOT/checkpoints/normal_opd")
sft_ckpt=$(latest_checkpoint "$ROOT/checkpoints/normal_sft_tokens")
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; \
    python -m evals.workspace_readout --base-ckpt '$BASE' --av-ckpt '$opd_ckpt' \
      --sidecar '$opd_ckpt' --jdir '$JDIR' --layers 20,32,42,54,62 \
      --modes raw,jac,shuffle --samples 2 --max-new-tokens 24 \
      --out '$ROOT/results/stage3/workspace_opd.json'" &
opd_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; \
    python -m evals.workspace_readout --base-ckpt '$BASE' --av-ckpt '$sft_ckpt' \
      --sidecar '$sft_ckpt' --jdir '$JDIR' --layers 20,32,42,54,62 \
      --modes raw,jac,shuffle --samples 2 --max-new-tokens 24 \
      --out '$ROOT/results/stage3/workspace_sft_tokens.json'" &
sft_pid=$!
wait "$opd_pid"
wait "$sft_pid"

# Large judge runs use the discounted, resumable Message Batches endpoint.
python scripts/merge_eval_json.py \
  --inputs "$ROOT"/results/stage3/normal_*_eval.json \
  --out "$ROOT/results/stage3/normal_quality_inputs.json"
python -m evals.judge_opd_quality \
  --input "$ROOT/results/stage3/normal_quality_inputs.json" \
  --out "$ROOT/results/stage3/normal_quality_judged.json"
python -m evals.judge_workspace_readouts \
  --input "$ROOT/results/stage3/workspace_opd.json" \
  --out "$ROOT/results/stage3/workspace_opd_judged.json"
python -m evals.judge_workspace_readouts \
  --input "$ROOT/results/stage3/workspace_sft_tokens.json" \
  --out "$ROOT/results/stage3/workspace_sft_tokens_judged.json"
echo STAGE3_DONE
