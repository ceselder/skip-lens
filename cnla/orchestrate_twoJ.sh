#!/bin/bash
# TWO-J arm (arm E) — the design the user specified.
#
#   TRAIN slots[d] = Jbar^(d)_{62->63} @ h62[p]     (source = penultimate)
#   TEST  slots[d] = Jbar^(d)_{42->63} @ h42[p]     (source = L42)
#
# Same estimator type, same target layer (the LAST block, where
# W_U.norm(h) IS the model's prediction — no approximation), and the ONLY
# train/test difference is how deep you source from. That makes the gap exactly
# the skip-lens variable, unlike arm A where train inputs were directional
# derivatives and test inputs were activation estimates (0.156 aligned).
#
# Both families come from ONE fit: the comb estimator takes gradients w.r.t.
# every source layer in the same backward passes, so L62->L63 and L42->L63 cost
# what one family costs. Training slots need NO new collection — h62[p] is
# already stored as act_L62 in the pass-1 shards, so building them is a matmul.
#
# Chain: wait for GPUs -> fit {42,62}->63 (sharded) -> merge -> sanity-check the
# 62->63 family (its only causal path to later positions is block 63's
# attention, so verify it is signal and not noise) -> build train parquet ->
# train -> ready for the judged eval with the 42->63 family.
set -u
cd /workspace/skip-lens
source /workspace/.keys.env
LOG=/workspace/logs/orchestrator_twoJ.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

OUT_DIR=/workspace/results/offset_jlens_last
NEED_MB=${NEED_MB:-150000}          # the fit retains a 21-block graph x dim_batch 32
N_PROMPTS=${N_PROMPTS:-360}
FINAL_T=/workspace/data/final/twoJ_multislot_train.parquet
FINAL_V=/workspace/data/final/twoJ_multislot_val.parquet
mkdir -p "$OUT_DIR" logs

free_mb(){ echo $(( $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1") - $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$1") )); }

if [ ! -f "$OUT_DIR/Jbar_L62_to_L63_off0.npy" ]; then
  log "waiting for 3 GPUs with >= ${NEED_MB}MB free"
  while true; do
    ready=(); for g in 0 1 2 3; do [ "$(free_mb $g)" -ge "$NEED_MB" ] && ready+=("$g"); done
    [ "${#ready[@]}" -ge 3 ] && break
    sleep 180
  done
  log "GPUs ready: ${ready[*]} — fitting {42,62} -> 63, $N_PROMPTS prompts"
  for i in 0 1 2; do
    g=${ready[$i]}
    ( source /workspace/venv/bin/activate
      SHARD=$i N_SHARDS=3 SRC="42,62" TGT=63 N_OFFSETS=16 COMB=32 SEQ_LEN=512 \
      DIM_BATCH=32 N_PROMPTS=$N_PROMPTS OUT_DIR="$OUT_DIR" \
      PYTHONPATH=/workspace/pylib:/workspace/skip-lens \
      CUDA_VISIBLE_DEVICES=$g setsid python remote_src/fit_offset_jlens.py \
        > "logs/twoJ_fit_shard${i}.log" 2>&1 < /dev/null & )
    log "launched fit shard $i on gpu $g"
    sleep 10
  done
  log "waiting for the 3 fit shards"
  while true; do
    n=0
    for i in 0 1 2; do
      grep -q "OFFSET FIT SHARD $i DONE" "logs/twoJ_fit_shard${i}.log" 2>/dev/null && n=$((n+1))
    done
    [ "$n" -eq 3 ] && break
    sleep 300
  done
  source /workspace/venv/bin/activate
  SRC="42,62" TGT=63 OUT_DIR="$OUT_DIR" \
    PYTHONPATH=/workspace/pylib:/workspace/skip-lens \
    python remote_src/merge_offset_fit.py 2>&1 | tee -a "$LOG"
fi

# ---- sanity: is the 62->63 family signal or noise? -------------------------
# h62[p] reaches position p+d ONLY through block 63's attention, so its
# per-offset transports are structurally much weaker than the 21-block
# 42->63 paths. If half-split cosine collapses at d>=1 the training slots
# would be noise and the arm is not worth training.
source /workspace/venv/bin/activate
python - <<'PY' 2>&1 | tee -a "$LOG"
import json
d = json.load(open("/workspace/results/offset_jlens_last/offset_fit_diagnostics.json"))
for src, rows in d["sources"].items():
    print(f"source L{src} -> L63:")
    print("  |J| by delta:      " + " ".join(f"{r['fro_norm']:7.3f}" for r in rows[:8]))
    print("  split-cos by delta:" + " ".join(f"{r['half_split_cosine']:7.3f}" for r in rows[:8]))
lo = [r["half_split_cosine"] for r in d["sources"]["62"][1:8]]
print("VERDICT:", "62->63 transports are SIGNAL (min split-cos %.3f over d=1..7)" % min(lo)
      if min(lo) > 0.5 else
      "62->63 transports look like NOISE at d>=1 (min split-cos %.3f) — "
      "training slots would be noise; reconsider the train-time source layer" % min(lo))
PY

log "building train parquet (slots = Jbar_{62->63} @ act_L62, from existing pass-1 shards)"
PYTHONPATH=/workspace/skip-lens python pretrain/build_twoJ_train.py \
  --raw-glob "/workspace/data/spans_raw/shard_[0-3].parquet" \
  --jbar-dir "$OUT_DIR" --src-layer 62 --tgt-layer 63 --k-slots 8 \
  --out-train "$FINAL_T" --out-val "$FINAL_V" 2>&1 | tee -a "$LOG"

NSTEPS=$(python -c "import pyarrow.parquet as pq;print(max(200,pq.ParquetFile('$FINAL_T').metadata.num_rows//64))")
g=""; while [ -z "$g" ]; do for c in 0 1 2 3; do [ "$(free_mb $c)" -ge 70000 ] && { g=$c; break; }; done; [ -z "$g" ] && sleep 120; done
log "training arm E (two-J) for $NSTEPS steps on gpu $g"
CUDA_VISIBLE_DEVICES=$g PYTHONPATH=/workspace/skip-lens setsid \
  python -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$FINAL_T" --sidecar "$FINAL_T" --n-slots 8 \
  --save-dir ckpts/multislot_armE_twoJ \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$NSTEPS" --batch-size 16 --gradient-accumulation-steps 4 \
  --save-every 500 --heldout-parquet "$FINAL_V" --heldout-rows 800 \
  --heldout-every 250 --sample-every 500 --n-samples 4 \
  --sample-max-new-tokens 48 \
  --wandb-project skiplens-multislot --wandb-name armE_twoJ_62to63 \
  > logs/train_armE.log 2>&1 < /dev/null &
log "arm E training launched — orchestrator exiting"
