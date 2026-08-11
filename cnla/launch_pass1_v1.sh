#!/bin/bash
# Pass-1 ON-POLICY span collection on the 4xB200 box (fresh corpus, both layers).
# Each row: L62 + L42 residuals at position p + a 16-token rollout the MODEL
# GENERATES from the prefix (temp 1.0 top-p 0.95) — on-policy per the design.
# activation_vector = act_L62 (arm-C convention); pass 2 reads act_L42.
#
# Usage: bash cnla/launch_pass1_v1.sh "<gpu list>" <docs-per-shard> <first-shard-idx>
#   e.g. bash cnla/launch_pass1_v1.sh "0" 8000 0        (shard 0 on gpu 0)
#        bash cnla/launch_pass1_v1.sh "1 2 3" 8000 1    (shards 1-3 after the fit)
set -u
GPUS=(${1:?gpu list})
DOCS=${2:-8000}
FIRST=${3:-0}
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens
CORPUS=/workspace/data/ffw_corpus_100k.parquet
mkdir -p /workspace/data/spans_raw logs
for i in $(seq 0 $((${#GPUS[@]}-1))); do
  SHARD=$((FIRST+i)); GPU=${GPUS[$i]}
  CUDA_VISIBLE_DEVICES=$GPU setsid python pretrain/collect_ao_data.py \
    --base-ckpt Qwen/Qwen3.6-27B \
    --corpus "$CORPUS" \
    --out /workspace/data/spans_raw/shard_${SHARD}.parquet \
    --layers 62 42 --n-docs $DOCS --doc-offset $((SHARD*DOCS)) \
    --positions-per-doc 8 --rollouts 1 --rollout-len 16 --seed $SHARD \
    > logs/pass1_shard${SHARD}.log 2>&1 < /dev/null &
  echo "launched pass1 shard $SHARD on gpu $GPU pid $!"
  sleep 10
done
echo "pass1 shards launched"
