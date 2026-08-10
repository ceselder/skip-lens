#!/bin/bash
#SBATCH --job-name=repeat_span4
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
DATA=$ROOT/data/repeat_scaled_all_25k_span4
CKPTS=$ROOT/checkpoints/repeat_scaled_all_25k_span4
RESULTS=$ROOT/results/repeat_scaled_all_25k_span4

source "$VENV/bin/activate"
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_HUB_OFFLINE=1
export PYTHONPATH=$SRC
mkdir -p "$CKPTS" "$RESULTS" "$ROOT/logs"
cd "$SRC"

python - <<'PY'
import pyarrow.parquet as pq

root = "/workspace-vast/celeste/skip-lens-opd/data/repeat_scaled_all_25k_span4"
for split, expected in (("train", 22504), ("val", 2496)):
    table = pq.read_table(
        f"{root}/{split}.parquet",
        columns=["target_ids", "continuation_ids", "span_tokens"],
    )
    assert table.num_rows == expected
    assert all(len(x) == 4 for x in table["target_ids"].to_pylist())
    assert all(len(x) == 4 for x in table["continuation_ids"].to_pylist())
    assert set(table["span_tokens"].to_pylist()) == {4}
print("four-token dataset validated")
PY

python -m nla.train_sft --mode av --base-ckpt "$BASE" \
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

python -m evals.opd_eval --base-ckpt "$BASE" --av-ckpt "$BEST" \
  --parquet "$DATA/val.parquet" --sidecar "$DATA/val.parquet" \
  --feed-col activation_vector --max-rows 1024 --max-new-tokens 4 --batch-size 8 \
  --out "$RESULTS/repeat_scaled_L62_eval.json"

python -m evals.repeat_token_rank_eval --base-ckpt "$BASE" --av-ckpt "$BEST" \
  --parquet "$DATA/val.parquet" --sidecar "$DATA/val.parquet" \
  --feed-cols activation_vector --max-rows 1024 --max-tokens 4 --batch-size 32 \
  --out "$RESULTS/repeat_token_ranks.json"

python scripts/plot_repeat_validation_curve.py \
  --metrics "$CKPTS/metrics.jsonl" --out-dir "$RESULTS"

echo REPEAT_SCALED_SPAN4_SLURM_DONE
