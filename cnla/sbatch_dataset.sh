#!/bin/bash
#SBATCH --job-name=cnla_ds
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=5:00:00
#SBATCH --output=/workspace-vast/celeste/cnla-skip-lens/logs/cnla_ds_%j.out
set -uo pipefail
cd /workspace-vast/celeste/cnla-skip-lens
export HF_HOME=/workspace-vast/pretrained_ckpts HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
V=/workspace-vast/celeste/oracle-lens/.venv/bin/python
echo "=== node $(hostname) gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Stage 1: build the raw collect parquet — L62 activation + 4 sampled completions
# @T=1.0 at high-entropy decision points (proven HF collector; checkpoints /200 docs).
$V pretrain/collect_ao_data.py --base-ckpt Qwen/Qwen3.6-27B \
  --corpus /workspace-vast/celeste/nla-data/finefineweb_100k.parquet \
  --out data/cnla/collect_L62.parquet --layer 62 \
  --rollouts 4 --rollout-len 16 --n-docs 1000 --positions-per-doc 5 --seed 0

# Stage 2: format into the "* bullet" text + human-readable preview
$V cnla/format_preview.py --in data/cnla/collect_L62.parquet --out data/cnla/cnla_L62 --n-preview 40
echo "DATASET_BUILD_DONE"
