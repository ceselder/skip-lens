#!/bin/bash
#SBATCH --job-name=skiplens_repeat_all
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
DATA=$ROOT/data/repeat_scaled_all_25k
CKPTS=$ROOT/checkpoints/repeat_scaled_all_25k
RESULTS=$ROOT/results/repeat_scaled_all_25k
JDIR=${JDIR:-/workspace-vast/celeste/multi-token-jlens-nla-lastlayer/results/jlens_official}

source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
mkdir -p "$DATA" "$CKPTS" "$RESULTS" "$ROOT/logs"
cd "$SRC"

# 1,563 updates * batch 16 = 25,008 example presentations, about 1.11 passes
# through 22,500 phrase-disjoint training rows and roughly 0.2M tokens.
# `all` covers attention, MLP, and architecture-specific DeltaNet projections.
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
  --wandb-name repeat_25kactivations_bs16_allmodules

python scripts/select_best_sft_checkpoint.py \
  --metrics "$CKPTS/metrics.jsonl" --checkpoint-dir "$CKPTS" \
  --out "$RESULTS/best_checkpoint.json"
BEST=$(python -c "import json; print(json.load(open('$RESULTS/best_checkpoint.json'))['checkpoint'])")

for feed in activation_vector act_L42; do
  if [ "$feed" = activation_vector ]; then label=L62; else label=L42; fi
  python -m evals.opd_eval --base-ckpt "$BASE" --av-ckpt "$BEST" \
    --parquet "$DATA/val.parquet" --sidecar "$DATA/val.parquet" \
    --feed-col "$feed" --max-rows 1024 --max-new-tokens 8 --batch-size 8 \
    --out "$RESULTS/repeat_scaled_${label}_eval.json"
done

# Raw L42 is the hypothesis test; L62 is its in-distribution control.
python -m evals.workspace_readout \
  --base-ckpt "$BASE" --av-ckpt "$BEST" --sidecar "$BEST" --jdir "$JDIR" \
  --layers 42,62 --modes raw --samples 2 --max-new-tokens 24 \
  --temperature 0.8 --seed 0 \
  --out "$RESULTS/workspace_raw_L42_L62.json"

echo REPEAT_SCALED_ALL_DONE
