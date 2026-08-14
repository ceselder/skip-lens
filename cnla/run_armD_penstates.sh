#!/bin/bash
# ARM D — the multi-slot generalization of the ONE recipe that works.
#
# Corrected-scale scoreboard (normalized 0-1, judge emits 0/1/2):
#   armC single-slot   0.561   trained on RAW h62, tested on pooled Jbar.h42
#   armFrozen diff     0.308   trained on chord transports
#   armA diff          0.293   trained on local JVP transports
#   armA per_offset    0.145   the design as specified
#
# The winner's distinguishing feature is not the slot count — it is WHAT the
# decoder was trained to read. armC trained on a real activation and is tested
# on an averaged-Jacobian ESTIMATE of that same kind of object (the J-bar audit
# measured cos(Jbar.h42, true h62) = +0.50). Every multi-slot arm so far trained
# on transports/derivatives instead, which are a different kind of object
# (cos 0.156 with the test-time vector).
#
# Arm D keeps armC's recipe and only adds slots:
#   TRAIN slots[d] = the REAL h62[p+d]        (forward-only collection)
#   TEST  slots[d] = Jbar^(d)_{42->62} @ h42[p]
#
# Prediction to check against: if the multi-slot format itself is fine and the
# earlier failures were about training on the wrong object, arm D should land
# near armC (0.561) rather than near the transport arms (~0.3).
set -u
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens

OUT=/workspace/data/spans_pen8
T=/workspace/data/final/pen8_multislot_train.parquet
V=/workspace/data/final/pen8_multislot_val.parquet
mkdir -p "$OUT" logs

# ---- collect the real penultimate states (forward-only, no autodiff) ----
for i in 0 1 2 3; do
  [ -f "$OUT/shard_${i}_jvp.parquet" ] && { echo "[armD] shard $i present"; continue; }
  G=${FORCE_GPU:-$(/workspace/gpu_wait.sh 65000)}
  echo "[armD] collecting shard $i on gpu $G"
  CUDA_VISIBLE_DEVICES=$G python pretrain/collect_pen_states.py \
    --in-shards "/workspace/data/spans_raw/shard_${i}.parquet" \
    --out-dir "$OUT" --batch-size 32 >> "logs/pen8_shard${i}.log" 2>&1 \
    || { echo "[armD] shard $i FAILED — see logs/pen8_shard${i}.log"; exit 1; }
done
echo "[armD] all real penultimate states collected"

# ---- finalize (same builder as the other arms; slots are already in the file) ----
if [ ! -f "$T" ]; then
  python pretrain/finalize_jvp_spans.py \
    --shards-glob "$OUT/shard_[0-3]_jvp.parquet" \
    --out-train "$T" --out-val "$V" --k-slots 8 || exit 1
fi

N=$(python -c "import pyarrow.parquet as pq; print(max(200, pq.ParquetFile('$T').metadata.num_rows // 64))")
G=${FORCE_GPU:-$(/workspace/gpu_wait.sh 70000)}
echo "[armD] training on gpu $G for $N steps"
CUDA_VISIBLE_DEVICES=$G python -m nla.train_sft --mode av \
  --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$T" --sidecar "$T" --n-slots 8 \
  --save-dir ckpts/multislot_armD_pen8 \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$N" --batch-size 16 --gradient-accumulation-steps 4 \
  --save-every 500 --heldout-parquet "$V" --heldout-rows 800 --heldout-every 250 \
  --sample-every 500 --n-samples 4 --sample-max-new-tokens 48 \
  --wandb-project skiplens-multislot --wandb-name armD_penstates_k8
echo "[armD] done — evaluate with --jbar-dir /workspace/results/offset_jlens --conditions per_offset,diff"
