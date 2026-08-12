#!/bin/bash
# Frozen-routing arm, end to end, autonomous.
#
# Motivation (measured, diag_frozen_routing.py): transports computed with
# content-dependent routing DETACHED (attention softmax, GatedDeltaNet gates,
# RMSNorm denominators, SiLU gates) align 2.4-3.1x better with the corpus-
# averaged Jbar family than ordinary local transports do (0.124 -> 0.304 at
# horizon 3). So training on frozen-routing transports shrinks the train/test
# mismatch from the TRAINING side, leaving the averaged-Jbar test-time readout
# completely unchanged.
#
# Chain: wait for GPUs -> pass 2 with --frozen-routing (4 shards, 4 GPUs)
#     -> finalize -> retrain arm A-frozen -> ready for the same judged eval.
#
# Reuses pass-1 shards 0-3 (raw spans): NO re-collection of rollouts needed.
set -u
cd /workspace/skip-lens
source /workspace/.keys.env
LOG=/workspace/logs/orchestrator_frozen.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

NEED_MB=${NEED_MB:-115000}      # free MB required on a GPU to host a worker
GPUS=(0 1 2 3)
OUT=/workspace/data/spans_frozen
FINAL_T=/workspace/data/final/frozen_multislot_train.parquet
FINAL_V=/workspace/data/final/frozen_multislot_val.parquet
mkdir -p "$OUT" logs

free_mb(){ # $1 = gpu index
  local used total
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1")
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1")
  echo $((total - used))
}

log "frozen arm: waiting for 4 GPUs with >= ${NEED_MB}MB free (audit agents are using them)"
while true; do
  ready=()
  for g in "${GPUS[@]}"; do
    [ "$(free_mb "$g")" -ge "$NEED_MB" ] && ready+=("$g")
  done
  [ "${#ready[@]}" -ge 4 ] && break
  sleep 180
done
log "GPUs ready: ${ready[*]}"

# ---- pass 2 with frozen routing: one shard per GPU ----
for i in 0 1 2 3; do
  g=${ready[$i]}
  if [ -f "$OUT/shard_${i}_jvp.parquet" ]; then log "shard $i already done"; continue; fi
  ( source /workspace/venv_jvp/bin/activate
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH=/workspace/skip-lens \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid \
    python pretrain/collect_jvp_transport.py \
      --in-shards "/workspace/data/spans_raw/shard_${i}.parquet" \
      --out-dir "$OUT" --worker 0 --n-workers 1 \
      --batch-size 64 --dtype bf16 --backend jvp --frozen-routing \
      --probe-frac 0.02 > "logs/pass2frozen_shard${i}.log" 2>&1 < /dev/null & )
  log "launched frozen pass-2 shard $i on gpu $g"
  sleep 10
done

log "waiting for all 4 frozen shards"
while true; do
  n=0
  for i in 0 1 2 3; do [ -f "$OUT/shard_${i}_jvp.parquet" ] && n=$((n+1)); done
  [ "$n" -eq 4 ] && break
  sleep 180
done
log "all frozen transports collected"

# ---- finalize ----
if [ ! -f "$FINAL_T" ]; then
  source /workspace/venv/bin/activate
  PYTHONPATH=/workspace/skip-lens python pretrain/finalize_jvp_spans.py \
    --shards-glob "$OUT/shard_[0-3]_jvp.parquet" \
    --out-train "$FINAL_T" --out-val "$FINAL_V" --k-slots 8 2>&1 | tee -a "$LOG"
fi

# ---- retrain (same recipe as arm A, only the transports differ) ----
source /workspace/venv/bin/activate
NSTEPS=$(python -c "import pyarrow.parquet as pq;print(max(200,pq.ParquetFile('$FINAL_T').metadata.num_rows//64))")
log "retraining arm A-frozen for $NSTEPS steps on gpu ${ready[0]}"
CUDA_VISIBLE_DEVICES=${ready[0]} PYTHONPATH=/workspace/skip-lens setsid \
  python -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$FINAL_T" --sidecar "$FINAL_T" --n-slots 8 \
  --save-dir ckpts/multislot_armA_frozen_k8 \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$NSTEPS" --batch-size 16 --gradient-accumulation-steps 4 \
  --save-every 500 --heldout-parquet "$FINAL_V" --heldout-rows 800 \
  --heldout-every 250 --sample-every 500 --n-samples 4 \
  --sample-max-new-tokens 48 \
  --wandb-project skiplens-multislot --wandb-name armA_frozen_k8 \
  > logs/train_armA_frozen.log 2>&1 < /dev/null &
log "training launched — orchestrator exiting"
