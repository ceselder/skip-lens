#!/bin/bash
# Per-offset Jacobian fit J^(delta)_{L42->L62}, delta=0..15, sharded across GPUs.
# Comb estimator with Rademacher tooth signs (jlens/offset_fitting.py); merge
# shards afterwards with remote_src/merge_offset_fit.py. ~510 prompts x 512 tok
# gives ~40k Jacobian samples per offset bucket (per-delta noise level roughly
# matching the 48-prompt pooled fits used previously, x8 margin).
# Usage: bash cnla/launch_offset_fit.sh "1 2 3"   (GPU ids; default "1 2 3")
set -u
GPUS=(${1:-"1 2 3"})
N=${#GPUS[@]}
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/pylib:/workspace/skip-lens
export N_PROMPTS=${N_PROMPTS:-510} N_OFFSETS=16 COMB=32 SEQ_LEN=512
export DIM_BATCH=${DIM_BATCH:-32}
export OUT_DIR=/workspace/results/offset_jlens
mkdir -p logs "$OUT_DIR"
for i in $(seq 0 $((N-1))); do
  GPU=${GPUS[$i]}
  SHARD=$i N_SHARDS=$N CUDA_VISIBLE_DEVICES=$GPU setsid \
    python remote_src/fit_offset_jlens.py \
    > logs/offset_fit_shard${i}.log 2>&1 < /dev/null &
  echo "launched fit shard $i/$N on gpu $GPU pid $!"
  sleep 8
done
echo "all offset-fit shards launched; logs/offset_fit_shard*.log"
