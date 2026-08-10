#!/bin/bash
#SBATCH --job-name=skiplens_fw_t1_half
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

# Corrected OPD-vs-SFT experiment:
#   shared first half: 10,000 SFT response tokens from T=1 FineWeb rollouts;
#   second half:       10,000 sampled reverse-KL OPD tokens OR 10,000 SFT tokens.
# Both arms use the same rows, warm checkpoint, optimizer family, LR, batch size,
# and 8-token horizon.  The OPD token mask clips its final batch exactly.

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
EXP=$ROOT/results/fineweb_t1_half
DATA=$ROOT/data/fineweb_t1_half
CKPTS=$ROOT/checkpoints/fineweb_t1_half
TOKEN_BUDGET=10000

source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
mkdir -p "$EXP" "$DATA" "$CKPTS" "$ROOT/logs"
cd "$SRC"

# One exact T=1.0, untruncated model continuation is the static SFT label for
# each activation. Uniform positions avoid an entropy-selection confound.
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  python -m pretrain.collect_ao_data \
    --base-ckpt "$BASE" \
    --corpus HuggingFaceFW/fineweb --corpus-config sample-10BT \
    --layers 62 42 --n-docs 700 --positions-per-doc 5 \
    --rollouts 1 --rollout-len 8 --temperature 1.0 --top-p 1.0 \
    --no-decision-points --seed 20260810 \
    --out "$DATA/collected.parquet"

python -m pretrain.finalize_opd_data \
  --collected "$DATA/collected.parquet" \
  --meta "$DATA/collected.parquet.meta.json" \
  --out-train "$DATA/train.parquet" --out-val "$DATA/val.parquet" \
  --max-target-tokens 8 --val-frac 0.10 --split-seed 20260810

# Create a zero-step adapter, then train on exact stored token IDs.  This avoids
# decode/re-tokenize drift in the general text SFT trainer.
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  python -m nla.train_sft --mode av --base-ckpt "$BASE" \
    --parquet "$DATA/train.parquet" --sidecar "$DATA/train.parquet" \
    --save-dir "$CKPTS/initial" --num-steps 0 --batch-size 2 \
    --use-lora --lora-r 64 --lora-alpha 16 --save-initial --no-wandb

INITIAL=$CKPTS/initial/iter_0000000

# 625 updates * batch 2 * 8 stored target IDs = exactly 10,000 shared SFT tokens.
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  python -m nla.train_opd --objective sft --base-ckpt "$BASE" \
    --av-ckpt "$INITIAL" --parquet "$DATA/train.parquet" \
    --sidecar "$DATA/train.parquet" --save-dir "$CKPTS/warm_sft_10k" \
    --num-steps 625 --lr-decay-steps 625 --batch-size 2 \
    --max-new-tokens 8 --max-optimized-tokens "$TOKEN_BUDGET" \
    --lr 3e-5 --min-lr 3e-5 --save-every 625 \
    --wandb-project skip-lens-opd --wandb-group fineweb-t1-half \
    --wandb-name fineweb_t1_shared_sft_10k

WARM=$CKPTS/warm_sft_10k/iter_0000625

train_arm() {
  local objective=$1
  local name=$2
  python -m nla.train_opd --objective "$objective" --base-ckpt "$BASE" \
    --av-ckpt "$WARM" --parquet "$DATA/train.parquet" \
    --sidecar "$DATA/train.parquet" --save-dir "$CKPTS/$name" \
    --num-steps 20000 --lr-decay-steps 20000 --batch-size 2 \
    --max-new-tokens 8 --max-optimized-tokens "$TOKEN_BUDGET" \
    --temperature 1.0 --lr 3e-5 --min-lr 3e-5 --save-every 500 \
    --wandb-project skip-lens-opd --wandb-group fineweb-t1-half \
    --wandb-name "fineweb_t1_${name}"
}
export -f train_arm
export ROOT SRC VENV BASE EXP DATA CKPTS TOKEN_BUDGET WARM HF_HOME HF_TOKEN_PATH PYTHONPATH

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc 'source /workspace-vast/celeste/.keys.env; source "$VENV/bin/activate"; cd "$SRC"; train_arm opd opd_10k' &
opd_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc 'source /workspace-vast/celeste/.keys.env; source "$VENV/bin/activate"; cd "$SRC"; train_arm sft sft_10k' &
sft_pid=$!
wait "$opd_pid"
wait "$sft_pid"

OPD=$(find "$CKPTS/opd_10k" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' | sort | tail -n 1)
SFT=$(find "$CKPTS/sft_10k" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' | sort | tail -n 1)
test -n "$OPD"
test -n "$SFT"

eval_arm() {
  local name=$1
  local checkpoint=$2
  local feed label
  for feed in activation_vector act_L42; do
    if [ "$feed" = activation_vector ]; then label=L62; else label=L42; fi
    python -m evals.opd_eval --base-ckpt "$BASE" --av-ckpt "$checkpoint" \
      --parquet "$DATA/val.parquet" --sidecar "$DATA/val.parquet" \
      --feed-col "$feed" --max-rows 256 --max-new-tokens 8 --batch-size 4 \
      --out "$EXP/fineweb_${name}_${label}_eval.json"
  done
}
export -f eval_arm
export OPD SFT

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc 'source /workspace-vast/celeste/.keys.env; source "$VENV/bin/activate"; cd "$SRC"; eval_arm warm "$WARM"; eval_arm opd "$OPD"' &
eval_a_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc 'source /workspace-vast/celeste/.keys.env; source "$VENV/bin/activate"; cd "$SRC"; eval_arm sft "$SFT"' &
eval_b_pid=$!
wait "$eval_a_pid"
wait "$eval_b_pid"

python scripts/merge_eval_json.py \
  --inputs "$EXP"/fineweb_*_eval.json \
  --out "$EXP/quality_inputs.json"
echo FINEWEB_T1_HALF_DONE
