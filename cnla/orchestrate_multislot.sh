#!/bin/bash
# Autonomous stage-chaining for the multi-slot experiment (runs in tmux on the
# box, checks every 5 min). Stages:
#   1. all 3 offset-fit shards done  -> merge Jbar matrices
#   2. fit merged                    -> pass-1 shards 1-3 on GPUs 1-3
#   3. each pass-1 shard done        -> pass-2 dvjp worker on that GPU
#   4. all pass-2 done               -> finalize both arms + launch training
#                                       (arm A gpu 0, arm C gpu 1)
# Idempotent: every stage guards on its output's existence. State in
# /workspace/logs/orchestrator.log.
set -u
cd /workspace/skip-lens
source /workspace/.keys.env
LOG=/workspace/logs/orchestrator.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

fit_done(){
  local n=0
  for s in 0 1 2; do
    grep -q "OFFSET FIT SHARD $s DONE" logs/offset_fit_shard${s}.log 2>/dev/null && n=$((n+1))
  done
  [ "$n" -eq 3 ]
}
pass1_done(){ [ -f "/workspace/data/spans_raw/shard_$1.parquet" ] && \
              grep -q "^wrote " logs/pass1_shard$1.log 2>/dev/null; }
pass2_done(){ [ -f "/workspace/data/spans_jvp/shard_$1_jvp.parquet" ]; }

MERGED=/workspace/results/offset_jlens/Jbar_L42_to_L62_offpooled.npy
TRAIN_LAUNCHED=/workspace/logs/.train_launched
declare -A P2_LAUNCHED

log "orchestrator up"
while true; do
  # stage 1: merge fit
  if [ ! -f "$MERGED" ] && fit_done; then
    log "fit shards done -> merging"
    source /workspace/venv/bin/activate
    PYTHONPATH=/workspace/pylib:/workspace/skip-lens \
      python remote_src/merge_offset_fit.py 2>&1 | tee -a "$LOG"
    deactivate
  fi
  # stage 2: pass-1 shards 1-3 once fit GPUs free
  if [ -f "$MERGED" ] && [ ! -f logs/pass1_shard1.log ]; then
    log "launching pass-1 shards 1-3 on GPUs 1-3"
    bash cnla/launch_pass1_v1.sh "1 2 3" 8000 1 2>&1 | tee -a "$LOG"
  fi
  # stage 3: pass-2 per finished pass-1 shard (same GPU index as shard)
  for s in 0 1 2 3; do
    if pass1_done $s && ! pass2_done $s && [ -z "${P2_LAUNCHED[$s]:-}" ]; then
      # GPU s is free the moment its pass-1 shard finished
      log "pass-1 shard $s done -> pass-2 dvjp worker on GPU $s"
      # jvp backend at batch 64: benchmarked 2026-08-12 — ~10x dvjp bs4
      # throughput; dvjp OOMs at bs>=16 (double-backward graph). jvp == dvjp
      # numerically (cos 1.000 all deltas, same-graph identity verified).
      ( source /workspace/venv_jvp/bin/activate
        CUDA_VISIBLE_DEVICES=$s PYTHONPATH=/workspace/skip-lens \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid \
        python pretrain/collect_jvp_transport.py \
          --in-shards "/workspace/data/spans_raw/shard_${s}.parquet" \
          --out-dir /workspace/data/spans_jvp \
          --worker 0 --n-workers 1 --batch-size 64 --dtype bf16 \
          --backend jvp --probe-frac 0.02 \
          > logs/pass2_shard${s}.log 2>&1 < /dev/null & )
      P2_LAUNCHED[$s]=1
    fi
  done
  # stage 4: finalize + train once all pass-2 shards exist
  if [ ! -f "$TRAIN_LAUNCHED" ] && pass2_done 0 && pass2_done 1 && \
     pass2_done 2 && pass2_done 3; then
    log "all pass-2 done -> finalize + train (armA gpu0, armC gpu1)"
    bash cnla/launch_multislot_train.sh 0 1 2>&1 | tee -a "$LOG"
    touch "$TRAIN_LAUNCHED"
    log "training launched — orchestrator exiting"
    break
  fi
  sleep 300
done
