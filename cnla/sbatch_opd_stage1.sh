#!/bin/bash
#SBATCH --job-name=skiplens_stage1
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
source /workspace-vast/celeste/.keys.env
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC
mkdir -p "$ROOT"/{data,checkpoints,results,logs}

# Separate Slurm steps assign one GPU to each lane. Do not set
# CUDA_VISIBLE_DEVICES manually; Slurm owns that mapping.
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "
    set -euo pipefail
    source /workspace-vast/celeste/.keys.env
    source $VENV/bin/activate
    export HF_HOME=/workspace-vast/pretrained_ckpts
    export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
    export PYTHONPATH=$SRC
    cd $SRC
    python -m pretrain.collect_ao_data \\
      --base-ckpt $BASE --corpus m-a-p/FineFineWeb-sample \\
      --layers 62 42 --n-docs 700 --positions-per-doc 5 \\
      --rollouts 1 --rollout-len 16 --out $ROOT/data/normal_collect.parquet
    python -m pretrain.finalize_opd_data \\
      --collected $ROOT/data/normal_collect.parquet \\
      --meta $ROOT/data/normal_collect.parquet.meta.json \\
      --out-train $ROOT/data/normal_train.parquet \\
      --out-val $ROOT/data/normal_val.parquet --max-target-tokens 16
    python -m nla.train_sft --mode av --base-ckpt $BASE \\
      --parquet $ROOT/data/normal_train.parquet \\
      --sidecar $ROOT/data/normal_train.parquet \\
      --heldout-parquet $ROOT/data/normal_val.parquet --heldout-every 100 \\
      --save-dir $ROOT/checkpoints/normal_warm \\
      --num-steps 200 --batch-size 4 --use-lora --lora-r 64 --lora-alpha 16 \\
      --lr 3e-5 --save-every 100 \\
      --wandb-project skip-lens-opd --wandb-group warmstart \\
      --wandb-name normal_16tok_warm
  " &
normal_pid=$!

srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "
    set -euo pipefail
    source /workspace-vast/celeste/.keys.env
    source $VENV/bin/activate
    export HF_HOME=/workspace-vast/pretrained_ckpts
    export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
    export PYTHONPATH=$SRC
    cd $SRC
    python -m pretrain.collect_repeat_data --base-model $BASE \\
      --out-train $ROOT/data/repeat_train.parquet \\
      --out-val $ROOT/data/repeat_val.parquet \\
      --n-phrases 700 --phrase-words 40 --positions-per-phrase 4 \\
      --max-span 16 --layers 42 62 --target-layer 62 --batch-size 8
    python -m nla.train_sft --mode av --base-ckpt $BASE \\
      --parquet $ROOT/data/repeat_train.parquet \\
      --sidecar $ROOT/data/repeat_train.parquet \\
      --heldout-parquet $ROOT/data/repeat_val.parquet --heldout-every 100 \\
      --save-dir $ROOT/checkpoints/repeat_warm \\
      --num-steps 200 --batch-size 4 --use-lora --lora-r 64 --lora-alpha 16 \\
      --lr 3e-5 --save-every 100 \\
      --wandb-project skip-lens-opd --wandb-group warmstart \\
      --wandb-name repeat_16tok_warm
  " &
repeat_pid=$!

wait "$normal_pid"
wait "$repeat_pid"
echo STAGE1_DONE
