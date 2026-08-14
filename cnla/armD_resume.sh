#!/bin/bash
# Resume arm D once the parallel shard collectors finish: skip collection,
# finalize, then train. (The original orchestrator collected sequentially; the
# remaining shards were relaunched in parallel on the training GPUs' spare
# memory, so its own collector was killed as a duplicate and it exited.)
set -u
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens

OUT=/workspace/data/spans_pen8
T=/workspace/data/final/pen8_multislot_train.parquet
V=/workspace/data/final/pen8_multislot_val.parquet

echo "[armD] waiting for all 4 real-penultimate-state shards"
while true; do
  n=0
  for i in 0 1 2 3; do [ -f "$OUT/shard_${i}_jvp.parquet" ] && n=$((n+1)); done
  [ "$n" -eq 4 ] && break
  sleep 120
done
echo "[armD] all shards present"

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
echo "done."
