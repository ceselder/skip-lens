#!/bin/bash
# Pass-2: JVP local-transport collection over the pass-1 shards, one worker/GPU.
# MUST run in the fla-free venv (venv_jvp) — forward-mode AD needs the
# pure-torch DeltaNet path. bf16 model unless SELFTEST said otherwise.
# Usage: bash cnla/launch_pass2_jvp.sh "0 1 2 3"
set -u
GPUS=(${1:-"0 1 2 3"})
N=${#GPUS[@]}
cd /workspace/skip-lens
source /workspace/venv_jvp/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens
mkdir -p /workspace/data/spans_jvp logs
for i in $(seq 0 $((N-1))); do
  GPU=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$GPU setsid python pretrain/collect_jvp_transport.py \
    --in-shards "/workspace/data/spans_raw/shard_*.parquet" \
    --out-dir /workspace/data/spans_jvp \
    --worker $i --n-workers $N --batch-size 8 --dtype bf16 --probe-frac 0.02 \
    > logs/pass2_worker${i}.log 2>&1 < /dev/null &
  echo "launched pass2 worker $i on gpu $GPU pid $!"
  sleep 8
done
echo "pass2 workers launched"
