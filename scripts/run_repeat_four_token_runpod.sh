#!/bin/bash
set -euo pipefail

RUN_ROOT=${RUN_ROOT:-/workspace/skip-lens-50k}
SRC=$RUN_ROOT/src
VENV=$RUN_ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
SOURCE_DATA=$RUN_ROOT/data/repeat_scaled_all_25k
DATA=$RUN_ROOT/data/repeat_scaled_all_25k_span4
CKPTS=$RUN_ROOT/checkpoints/repeat_scaled_all_25k_span4
RESULTS=$RUN_ROOT/results/repeat_scaled_all_25k_span4

source "$VENV/bin/activate"
export HF_HOME=/workspace/.hf_home
export HF_HUB_OFFLINE=1
export PYTHONPATH=$SRC
mkdir -p "$DATA" "$CKPTS" "$RESULTS" "$RUN_ROOT/logs"
cd "$SRC"

for split in train val; do
  python -m pretrain.truncate_repeat_targets \
    --input "$SOURCE_DATA/$split.parquet" \
    --out "$DATA/$split.parquet" --tokens 4 --base-ckpt "$BASE"
done

python - <<'PY'
import os
import pyarrow.parquet as pq

root = "/workspace/skip-lens-50k/data/repeat_scaled_all_25k_span4"
for split, expected in (("train", 22504), ("val", 2496)):
    path = f"{root}/{split}.parquet"
    table = pq.read_table(path, columns=["target_ids", "continuation_ids", "span_tokens"])
    assert table.num_rows == expected, (split, table.num_rows)
    assert all(len(x) == 4 for x in table["target_ids"].to_pylist())
    assert all(len(x) == 4 for x in table["continuation_ids"].to_pylist())
    assert set(table["span_tokens"].to_pylist()) == {4}
print("four-token dataset validated")
PY

CUDA_VISIBLE_DEVICES=${GPU:-0} python -m nla.train_sft --mode av --base-ckpt "$BASE" \
  --parquet "$DATA/train.parquet" --sidecar "$DATA/train.parquet" \
  --heldout-parquet "$DATA/val.parquet" --heldout-rows 1000 \
  --heldout-every 250 --save-dir "$CKPTS" \
  --num-steps 1563 --batch-size 16 --gradient-accumulation-steps 1 \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --no-append-response-eos \
  --lr 3e-5 --min-lr 2e-6 --lr-warmup-steps 100 \
  --save-every 250 --sample-every 0 \
  --wandb-project skip-lens-opd --wandb-group repeat-scaled-all \
  --wandb-name repeat_25kactivations_span4_bs16_allmodules

python scripts/select_best_sft_checkpoint.py \
  --metrics "$CKPTS/metrics.jsonl" --checkpoint-dir "$CKPTS" \
  --out "$RESULTS/best_checkpoint.json"
BEST=$(python -c "import json; print(json.load(open('$RESULTS/best_checkpoint.json'))['checkpoint'])")

CUDA_VISIBLE_DEVICES=${GPU:-0} python -m evals.opd_eval --base-ckpt "$BASE" \
  --av-ckpt "$BEST" --parquet "$DATA/val.parquet" --sidecar "$DATA/val.parquet" \
  --feed-col activation_vector --max-rows 1024 --max-new-tokens 4 --batch-size 8 \
  --out "$RESULTS/repeat_scaled_L62_eval.json"

CUDA_VISIBLE_DEVICES=${GPU:-0} python -m evals.repeat_token_rank_eval --base-ckpt "$BASE" \
  --av-ckpt "$BEST" --parquet "$DATA/val.parquet" --sidecar "$DATA/val.parquet" \
  --feed-cols activation_vector --max-rows 1024 --max-tokens 4 --batch-size 32 \
  --out "$RESULTS/repeat_token_ranks.json"

python scripts/plot_repeat_validation_curve.py \
  --metrics "$CKPTS/metrics.jsonl" --out-dir "$RESULTS"

echo REPEAT_SCALED_SPAN4_RUNPOD_DONE
