#!/bin/bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/workspace/skip-lens-50k}
SRC=$RUN_ROOT/src
VENV=$RUN_ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
CKPTS=$RUN_ROOT/checkpoints/repeat_scaled_all_25k
RESULTS=$RUN_ROOT/results/repeat_scaled_all_25k
EMPTY_JDIR=$RUN_ROOT/empty_jlens
PIPELINE_PID_FILE=$RUN_ROOT/logs/repeat_scaled_pipeline.pid

source /workspace/.keys.env
source "$VENV/bin/activate"
export HF_HOME=/workspace/.hf_home
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONPATH=$SRC
mkdir -p "$RESULTS/workspace_trajectory" "$EMPTY_JDIR"
cd "$SRC"

# Avoid contending with training or its best-checkpoint evaluations.
pipeline_pid=$(cat "$PIPELINE_PID_FILE")
while kill -0 "$pipeline_pid" 2>/dev/null; do
  sleep 60
done
test -d "$CKPTS/iter_0001563"

python scripts/plot_repeat_validation_curve.py \
  --metrics "$CKPTS/metrics.jsonl" --out-dir "$RESULTS"

run_workspace() {
  local checkpoint=$1
  local gpu=$2
  local name step
  name=$(basename "$checkpoint")
  step=${name#iter_}
  CUDA_VISIBLE_DEVICES=$gpu python -m evals.workspace_readout \
    --base-ckpt "$BASE" --av-ckpt "$checkpoint" --sidecar "$checkpoint" \
    --jdir "$EMPTY_JDIR" --layers 42,62 --modes raw \
    --samples 2 --max-new-tokens 24 --temperature 0.8 --seed 0 \
    --out "$RESULTS/workspace_trajectory/workspace_step_${step}.json" \
    >"$RUN_ROOT/logs/workspace_step_${step}.log" 2>&1
}

mapfile -t checkpoints < <(find "$CKPTS" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' | sort)
for start in $(seq 0 4 $((${#checkpoints[@]} - 1))); do
  pids=()
  for offset in 0 1 2 3; do
    idx=$((start + offset))
    if [ "$idx" -ge "${#checkpoints[@]}" ]; then
      break
    fi
    run_workspace "${checkpoints[$idx]}" "$offset" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
done

python scripts/merge_workspace_trajectory.py \
  --inputs "$RESULTS"/workspace_trajectory/workspace_step_*.json \
  --out "$RESULTS/workspace_trajectory.json"
echo REPEAT_WORKSPACE_TRAJECTORY_DONE
