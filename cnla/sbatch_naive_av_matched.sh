#!/bin/bash
#SBATCH --job-name=nav_match
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/celeste/cnla-skip-lens/logs/nav_match_%j.out
set -uo pipefail
cd /workspace-vast/celeste/cnla-skip-lens
export HF_HOME=/workspace-vast/pretrained_ckpts HF_HUB_OFFLINE=1 PYTHONPATH=$PWD WANDB_MODE=disabled
V=/workspace-vast/celeste/oracle-lens/.venv/bin/python
# DATA-MATCHED naive future-lens: SFT on the SAME collect data as the CNLA AV, but response =
# a SINGLE continuation (rollouts[0]) instead of the 4-bullet block. So futurelens-init vs
# CNLA-init differ ONLY in output format, everything else identical.
$V pretrain/finalize_naive_data.py \
  --labeled data/cnla/collect_L62.parquet \
  --meta data/cnla/collect_L62.parquet.meta.json \
  --out-train data/cnla/navmatch_train.parquet --out-val data/cnla/navmatch_val.parquet \
  --rollout-idx 0
$V -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
  --parquet data/cnla/navmatch_train.parquet --sidecar data/cnla/navmatch_train.parquet \
  --save-dir ckpts/naive_av_matched_L62 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps 300 --batch-size 16 --gradient-accumulation-steps 4 --save-every 150
echo "NAV_MATCH_DONE"
