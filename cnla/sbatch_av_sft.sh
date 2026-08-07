#!/bin/bash
#SBATCH --job-name=cnla_av
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/celeste/cnla-skip-lens/logs/cnla_av_%j.out
set -uo pipefail
cd /workspace-vast/celeste/cnla-skip-lens
export HF_HOME=/workspace-vast/pretrained_ckpts HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
V=/workspace-vast/celeste/oracle-lens/.venv/bin/python
echo "=== node $(hostname) ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# 1. AV-SFT data: target = the full 4-bullet doc (doc-disjoint train/val + sidecar)
$V cnla/finalize_cnla_av.py --data data/cnla/cnla_L62.parquet \
  --meta data/cnla/collect_L62.parquet.meta.json \
  --out-train data/cnla/av_train.parquet --out-val data/cnla/av_val.parquet

# 2. reward whitener from the L62 activations (used later by CNLA-RL)
$V cnla/fit_whitener.py --data data/cnla/cnla_L62.parquet --out data/cnla/whitener.pt

# 3. AV-SFT: LoRA r64/a16 (rsLoRA hardcoded), all modules, Karvonen inject at block 1,
#    CE on the 4-bullet response. Seeds the RL policy's format.
$V -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
  --parquet data/cnla/av_train.parquet --sidecar data/cnla/av_train.parquet \
  --save-dir ckpts/cnla_av_L62 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps 300 --batch-size 16 --gradient-accumulation-steps 4 --save-every 150
echo "AV_SFT_DONE"
