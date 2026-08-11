#!/bin/bash
#SBATCH --job-name=fl_big_pt
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=10:00:00
#SBATCH --output=/workspace-vast/celeste/cnla-skip-lens/logs/fl_big_pt_%j.out
set -uo pipefail
cd /workspace-vast/celeste/cnla-skip-lens
export HF_HOME=/workspace-vast/pretrained_ckpts HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
source /workspace-vast/celeste/.keys.env   # wandb on by default now
V=/workspace-vast/celeste/oracle-lens/.venv/bin/python
# BIG futurelens PRETRAINING (AV-SFT) — same recipe as av_flv4, just many more steps, on the
# full 50,925-row ae_L62 corpus. Checkpoints every 500 steps = scaling points to eval the
# skip-lens suite on ("does more pretraining -> better skip-lens performance?").
DATA=/workspace-vast/celeste/multi-token-jlens-nla-lastlayer/data/ae_L62/train.parquet
STEPS=${STEPS:-5000}; SAVE=${SAVE:-500}
echo "=== node $(hostname) | BIG futurelens pretrain | data=$DATA | steps=$STEPS save-every=$SAVE ==="
$V -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$DATA" --sidecar "$DATA" \
  --save-dir ckpts/futurelens_big_L62 \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$STEPS" --batch-size 16 --gradient-accumulation-steps 4 --save-every "$SAVE" \
  --wandb-project futurelens-pretrain-scaling --wandb-group scaling --wandb-name futurelens_big \
  --wandb-tags pretrain,scaling,L62
echo "FL_BIG_PT_DONE"
