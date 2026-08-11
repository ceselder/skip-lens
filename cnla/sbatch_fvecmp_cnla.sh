#!/bin/bash
#SBATCH --job-name=fvecmp_cnla
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/celeste/cnla-skip-lens/logs/fvecmp_cnla_%j.out
set -uo pipefail
cd /workspace-vast/celeste/cnla-skip-lens
export HF_HOME=/workspace-vast/pretrained_ckpts HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /workspace-vast/celeste/.keys.env
V=/workspace-vast/celeste/oracle-lens/.venv/bin/python
# === CNLA (BULLETS) ARM of the data-matched FVE comparison ===
# compositional 4-bullet init; compositional leave-one-out FVE RL with a FROZEN AR,
# on the SAME activation set (av_train) + SAME AR as the futurelens arm. 8x128 on 2 GPUs.
AV=$(ls -d ckpts/cnla_av_L62/iter_* 2>/dev/null | sort -V | tail -1)
AR=/workspace-vast/celeste/easynla-ae/ckpts/ar_L62/iter_0001500
SIDE=/workspace-vast/celeste/multi-token-jlens-nla-lastlayer/data/ar_L62/train.parquet
STEPS=${STEPS:-150}; BP=${BP:-8}; GS=${GS:-128}; EVERY=${EVERY:-50}; KL=${KL:-0.1}
echo "=== node $(hostname) | CNLA ARM | AV=$AV | bp=$BP gs=$GS (=$((BP*GS))/step) ==="
$V -m cnla.train_cnla_rl \
  --base-ckpt Qwen/Qwen3.6-27B --quant none --device-map auto \
  --av-ckpt "$AV" --ar-ckpt "$AR" \
  --sidecar "$SIDE" --whitener data/cnla/whitener.pt \
  --rl-parquet data/cnla/av_train.parquet \
  --no-train-critic \
  --num-steps "$STEPS" --batch-prompts "$BP" --group-size "$GS" --logp-micro-batch 2 \
  --max-new-tokens 110 --temperature 1.0 \
  --kl-beta "$KL" --lr 3e-5 --save-every 25 \
  --eval-every "$EVERY" --text-judges-every "$EVERY" --evals base_fve text_judges \
  --wandb-project cnla-fve-compare --wandb-group fve-compare-datamatched \
  --wandb-name cnla_arm --wandb-tags data-matched,frozen-ar,8x128 \
  --save-dir ckpts/fvecmp_cnla_L62
echo "FVECMP_CNLA_DONE"
