#!/bin/bash
#SBATCH --job-name=fl_datagen
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/workspace-vast/celeste/cnla-skip-lens/logs/fl_datagen_%A_%a.out
set -uo pipefail
cd /workspace-vast/celeste/cnla-skip-lens
export HF_HOME=/workspace-vast/pretrained_ckpts HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
V=/workspace-vast/celeste/oracle-lens/.venv/bin/python
# SCALE THE DATA (not the epochs): sharded collection of a big FineFineWeb slice -> ~500k
# (activation, continuation) pairs. 6 shards x 16667 docs x ~5 positions. rollouts=2 (naive
# future-lens only uses rollouts[0]); the 16-rollout dense sample was a CNLA-only thing.
NDOCS=16667
OFFSET=$((SLURM_ARRAY_TASK_ID * NDOCS))
$V pretrain/collect_ao_data.py \
  --base-ckpt Qwen/Qwen3.6-27B \
  --corpus /workspace-vast/celeste/nla-data/finefineweb_xl.parquet \
  --out data/fl_big/shard_${SLURM_ARRAY_TASK_ID}.parquet \
  --layer 62 --n-docs $NDOCS --doc-offset $OFFSET \
  --positions-per-doc 5 --rollouts 2 --rollout-len 16 --seed $SLURM_ARRAY_TASK_ID
echo "DATAGEN_SHARD_${SLURM_ARRAY_TASK_ID}_DONE"
