#!/bin/bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/workspace/skip-lens-50k}
SRC=$RUN_ROOT/src
VENV=$RUN_ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
DATA=$RUN_ROOT/data/repeat_scaled_all_25k
CKPTS=$RUN_ROOT/checkpoints/repeat_scaled_all_25k
RESULTS=$RUN_ROOT/results/repeat_scaled_all_25k
EMPTY_JDIR=$RUN_ROOT/empty_jlens

source /workspace/.keys.env
source "$VENV/bin/activate"
export HF_HOME=/workspace/.hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH=$SRC
export REPEAT_DATA=$DATA
mkdir -p "$DATA" "$CKPTS" "$RESULTS" "$EMPTY_JDIR" "$RUN_ROOT/logs"
cd "$SRC"

phrase_counts=(1563 1563 1562 1562)
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$shard python -m pretrain.collect_repeat_data \
    --base-model "$BASE" \
    --out-train "$DATA/shard_${shard}_train.parquet" \
    --out-val "$DATA/shard_${shard}_val.parquet" \
    --n-phrases "${phrase_counts[$shard]}" \
    --phrase-words 40 --positions-per-phrase 4 --max-span 16 \
    --layers 42 62 --target-layer 62 --batch-size 16 \
    >"$RUN_ROOT/logs/repeat_collect_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

python -m pretrain.merge_repeat_shards \
  --inputs "$DATA"/shard_*_train.parquet --out "$DATA/train.parquet"
python -m pretrain.merge_repeat_shards \
  --inputs "$DATA"/shard_*_val.parquet --out "$DATA/val.parquet"

python - <<'PY'
import os
import pyarrow.parquet as pq

root = os.environ["REPEAT_DATA"]
train = set(pq.read_table(root + "/train.parquet", columns=["doc_id"])["doc_id"].to_pylist())
val = set(pq.read_table(root + "/val.parquet", columns=["doc_id"])["doc_id"].to_pylist())
if train & val:
    raise RuntimeError("phrase leakage across merged train/validation split")
train_rows = pq.read_metadata(root + "/train.parquet").num_rows
val_rows = pq.read_metadata(root + "/val.parquet").num_rows
if train_rows + val_rows != 25000:
    raise RuntimeError(f"expected exactly 25,000 activations, found {train_rows + val_rows}")
print(f"merged rows: train={train_rows} val={val_rows}; phrase overlap=0")
PY

CUDA_VISIBLE_DEVICES=0 python -m nla.train_sft --mode av --base-ckpt "$BASE" \
  --parquet "$DATA/train.parquet" --sidecar "$DATA/train.parquet" \
  --heldout-parquet "$DATA/val.parquet" --heldout-rows 1000 \
  --heldout-every 250 --save-dir "$CKPTS" \
  --num-steps 1563 --batch-size 16 --gradient-accumulation-steps 1 \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --lr 3e-5 --min-lr 2e-6 --lr-warmup-steps 100 \
  --save-every 250 --sample-every 0 \
  --wandb-project skip-lens-opd --wandb-group repeat-scaled-all \
  --wandb-name repeat_25kactivations_bs16_allmodules

python scripts/select_best_sft_checkpoint.py \
  --metrics "$CKPTS/metrics.jsonl" --checkpoint-dir "$CKPTS" \
  --out "$RESULTS/best_checkpoint.json"
BEST=$(python -c "import json; print(json.load(open('$RESULTS/best_checkpoint.json'))['checkpoint'])")

for feed in activation_vector act_L42; do
  if [ "$feed" = activation_vector ]; then label=L62; else label=L42; fi
  CUDA_VISIBLE_DEVICES=0 python -m evals.opd_eval --base-ckpt "$BASE" \
    --av-ckpt "$BEST" --parquet "$DATA/val.parquet" \
    --sidecar "$DATA/val.parquet" --feed-col "$feed" \
    --max-rows 1024 --max-new-tokens 16 --batch-size 8 \
    --out "$RESULTS/repeat_scaled_${label}_eval.json"
done

CUDA_VISIBLE_DEVICES=0 python -m evals.workspace_readout \
  --base-ckpt "$BASE" --av-ckpt "$BEST" --sidecar "$BEST" \
  --jdir "$EMPTY_JDIR" --layers 42,62 --modes raw \
  --samples 2 --max-new-tokens 24 --temperature 0.8 --seed 0 \
  --out "$RESULTS/workspace_raw_L42_L62.json"

echo REPEAT_SCALED_RUNPOD_DONE
