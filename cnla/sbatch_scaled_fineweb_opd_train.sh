#!/bin/bash
#SBATCH --job-name=skiplens_opd_scale_train
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

# Large-batch T=1 FineWeb experiment. Each optimizer update averages 256
# independently sampled activation/trajectory pairs (physical 32 × accumulate 8).

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
DATA=$ROOT/data/fineweb_t1_scaled_100k
CKPTS=$ROOT/checkpoints/fineweb_t1_scaled_100k
RESULTS=$ROOT/results/fineweb_t1_scaled_100k
PHYSICAL_BATCH=${PHYSICAL_BATCH:-32}
ACCUMULATION=${ACCUMULATION:-8}
WARM_UPDATES=175
STAGE2_UPDATES=500
WARM_TOKEN_BUDGET=358400
STAGE2_TOKEN_BUDGET=1024000

source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
mkdir -p "$DATA" "$CKPTS" "$RESULTS/evals" "$ROOT/logs"
cd "$SRC"

mapfile -t SHARDS < <(find "$DATA/shards" -maxdepth 1 -name 'collected_*.parquet' | sort)
if [ "${#SHARDS[@]}" -ne 8 ]; then
  echo "expected 8 collection shards, found ${#SHARDS[@]}" >&2
  exit 2
fi

python -m pretrain.merge_ao_shards --inputs "${SHARDS[@]}" \
  --out "$DATA/collected.parquet"
python -m pretrain.finalize_opd_data \
  --collected "$DATA/collected.parquet" \
  --meta "$DATA/collected.parquet.meta.json" \
  --out-train "$DATA/train.parquet" --out-val "$DATA/val.parquet" \
  --max-target-tokens 8 --val-frac 0.05 --split-seed 20260813
python -m pretrain.split_opd_phases \
  --input "$DATA/train.parquet" \
  --out-warm "$DATA/warm_train.parquet" \
  --out-stage2 "$DATA/stage2_train.parquet" \
  --rows-per-phase 45000 --seed 20260813

# Fresh rank-64 rsLoRA, followed by almost exactly one pass over 45k warm rows.
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  python -m nla.train_sft --mode av --base-ckpt "$BASE" \
    --parquet "$DATA/warm_train.parquet" --sidecar "$DATA/warm_train.parquet" \
    --save-dir "$CKPTS/initial" --num-steps 0 --batch-size 2 \
    --use-lora --lora-r 64 --lora-alpha 16 --save-initial --no-wandb

INITIAL=$CKPTS/initial/iter_0000000
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  python -m nla.train_opd --objective sft --base-ckpt "$BASE" \
    --av-ckpt "$INITIAL" --parquet "$DATA/warm_train.parquet" \
    --sidecar "$DATA/warm_train.parquet" --save-dir "$CKPTS/warm_sft" \
    --num-steps "$WARM_UPDATES" --lr-decay-steps "$WARM_UPDATES" \
    --batch-size "$PHYSICAL_BATCH" \
    --gradient-accumulation-steps "$ACCUMULATION" \
    --max-new-tokens 8 --max-optimized-tokens "$WARM_TOKEN_BUDGET" \
    --lr 3e-5 --min-lr 3e-6 --save-every 175 \
    --wandb-project skip-lens-opd --wandb-group fineweb-opd-scaled \
    --wandb-name fineweb_t1_scaled_shared_sft_warm

WARM=$CKPTS/warm_sft/iter_0000175
test -s "$WARM/adapter_model.safetensors"

train_arm() {
  local objective=$1
  local name=$2
  python -m nla.train_opd --objective "$objective" --base-ckpt "$BASE" \
    --av-ckpt "$WARM" --parquet "$DATA/stage2_train.parquet" \
    --sidecar "$DATA/stage2_train.parquet" --save-dir "$CKPTS/$name" \
    --num-steps "$STAGE2_UPDATES" --lr-decay-steps "$STAGE2_UPDATES" \
    --batch-size "$PHYSICAL_BATCH" \
    --gradient-accumulation-steps "$ACCUMULATION" \
    --max-new-tokens 8 --max-optimized-tokens "$STAGE2_TOKEN_BUDGET" \
    --temperature 1.0 --lr 3e-5 --min-lr 3e-6 --save-every 50 \
    --wandb-project skip-lens-opd --wandb-group fineweb-opd-scaled \
    --wandb-name "fineweb_t1_scaled_${name}"
}
export -f train_arm
export ROOT SRC VENV BASE DATA CKPTS RESULTS PHYSICAL_BATCH ACCUMULATION
export STAGE2_UPDATES STAGE2_TOKEN_BUDGET WARM HF_HOME HF_TOKEN_PATH PYTHONPATH

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc 'source /workspace-vast/celeste/.keys.env; source "$VENV/bin/activate"; cd "$SRC"; train_arm opd opd' &
opd_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc 'source /workspace-vast/celeste/.keys.env; source "$VENV/bin/activate"; cd "$SRC"; train_arm sft sft_matched' &
sft_pid=$!
wait "$opd_pid"
wait "$sft_pid"

eval_checkpoints() {
  local arm=$1
  local feed checkpoint step label
  for checkpoint in "$CKPTS/$arm"/iter_*; do
    step=${checkpoint##*/iter_}
    for feed in activation_vector act_L42; do
      if [ "$feed" = activation_vector ]; then label=L62; else label=L42; fi
      python -m evals.opd_eval --base-ckpt "$BASE" --av-ckpt "$checkpoint" \
        --parquet "$DATA/val.parquet" --sidecar "$DATA/val.parquet" \
        --feed-col "$feed" --max-rows 512 --max-new-tokens 8 --batch-size 8 \
        --out "$RESULTS/evals/${arm}_${step}_${label}.json"
    done
  done
}
export -f eval_checkpoints

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc 'source /workspace-vast/celeste/.keys.env; source "$VENV/bin/activate"; cd "$SRC"; eval_checkpoints opd' &
eval_opd_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc 'source /workspace-vast/celeste/.keys.env; source "$VENV/bin/activate"; cd "$SRC"; eval_checkpoints sft_matched' &
eval_sft_pid=$!
wait "$eval_opd_pid"
wait "$eval_sft_pid"

echo "SCALED_FINEWEB_OPD_TRAIN_EVAL_DONE"
