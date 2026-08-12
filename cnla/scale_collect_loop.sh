#!/bin/bash
# Continuous scale collection on one GPU: for each "shard:doc_offset:n_docs"
# item, run pass-1 (on-policy spans) then pass-2 (jvp transports), then move
# to the next. Idempotent (skips existing outputs). Optionally waits for a
# file first (e.g. the GPU's current pass-2 output) before starting.
# Usage: bash cnla/scale_collect_loop.sh <gpu> "s5:35000:8000 s7:51000:8000" [wait_file]
set -u
GPU=$1; ITEMS=($2); WAIT_FILE=${3:-}
cd /workspace/skip-lens
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens
LOG=logs/scale_gpu${GPU}.log
log(){ echo "[$(date +%H:%M:%S)] scale-gpu$GPU: $*" | tee -a "$LOG"; }

if [ -n "$WAIT_FILE" ]; then
  log "waiting for $WAIT_FILE"
  until [ -f "$WAIT_FILE" ]; do sleep 120; done
fi
for item in "${ITEMS[@]}"; do
  IFS=: read -r NAME OFF NDOCS <<< "$item"
  N=${NAME#s}
  RAW=/workspace/data/spans_raw/shard_${N}.parquet
  JVP=/workspace/data/spans_jvp/shard_${N}_jvp.parquet
  if [ ! -f "$RAW" ]; then
    log "pass-1 shard $N (docs $OFF..$((OFF+NDOCS)))"
    ( source /workspace/venv/bin/activate
      CUDA_VISIBLE_DEVICES=$GPU python pretrain/collect_ao_data.py \
        --base-ckpt Qwen/Qwen3.6-27B \
        --corpus /workspace/data/ffw_corpus_100k.parquet \
        --out "$RAW" --layers 62 42 --n-docs "$NDOCS" --doc-offset "$OFF" \
        --positions-per-doc 8 --rollouts 1 --rollout-len 16 --seed "$N" \
        >> "$LOG" 2>&1 )
  fi
  if [ -f "$RAW" ] && [ ! -f "$JVP" ]; then
    log "pass-2 shard $N"
    ( source /workspace/venv_jvp/bin/activate
      CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      python pretrain/collect_jvp_transport.py \
        --in-shards "$RAW" --out-dir /workspace/data/spans_jvp \
        --worker 0 --n-workers 1 --batch-size 64 --dtype bf16 --backend jvp \
        --probe-frac 0.02 >> "$LOG" 2>&1 )
  fi
  log "shard $N complete"
done
log "all items done"
