#!/bin/bash
# ARM F — the MATCHED CONTROL. Train on exactly the vectors the eval feeds:
#   slots[d] = Jbar^(d)_{42->62} @ h42[p]   at BOTH train and test time.
#
# Why this control matters now: every mechanistic story for the multi-slot
# readout's failure has assumed the train/test mismatch is the binding
# constraint, and all four have been falsified by their own predicted
# experiments (centering 0.218, deflation 0.236, keep0-deflate 0.138,
# keep0-gs 0.463, vs per_offset 0.289 and diff 0.586). If a decoder trained on
# PRECISELY its test-time inputs also fails to produce workspace readouts, then
# the mismatch was never the limit — the multi-slot averaged-Jacobian format is.
# This is the ceiling for arm A's test protocol.
#
# Note this trains on the averaged family, which the user deliberately excluded
# as a DESIGN (the averaging must stay a test-time projection). It is run here
# only as a diagnostic ceiling, not as a proposed lens.
set -u
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens

T=/workspace/data/final/armF_matched_train.parquet
V=/workspace/data/final/armF_matched_val.parquet

if [ ! -f "$T" ]; then
  G=${FORCE_GPU:-$(/workspace/gpu_wait.sh 20000)}
  echo "[armF] building matched train data on gpu $G"
  CUDA_VISIBLE_DEVICES=$G python pretrain/build_twoJ_train.py \
    --raw-glob "/workspace/data/spans_raw/shard_[0-3].parquet" \
    --jbar-dir /workspace/results/offset_jlens \
    --src-layer 42 --src-col act_L42 --tgt-layer 62 --k-slots 8 \
    --out-train "$T" --out-val "$V" || exit 1
fi

N=$(python -c "import pyarrow.parquet as pq; print(max(200, pq.ParquetFile('$T').metadata.num_rows // 64))")
G=${FORCE_GPU:-$(/workspace/gpu_wait.sh 70000)}
echo "[armF] training on gpu $G for $N steps"
CUDA_VISIBLE_DEVICES=$G python -m nla.train_sft --mode av \
  --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$T" --sidecar "$T" --n-slots 8 \
  --save-dir ckpts/multislot_armF_matched \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$N" --batch-size 16 --gradient-accumulation-steps 4 \
  --save-every 500 --heldout-parquet "$V" --heldout-rows 800 --heldout-every 250 \
  --sample-every 500 --n-samples 4 --sample-max-new-tokens 48 \
  --wandb-project skiplens-multislot --wandb-name armF_matched_42to62
echo "[armF] done"
