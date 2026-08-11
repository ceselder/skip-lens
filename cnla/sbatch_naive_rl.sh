#!/bin/bash
#SBATCH --job-name=naive_rl
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=14:00:00
#SBATCH --output=/workspace-vast/celeste/cnla-skip-lens/logs/naive_rl_%j.out
set -uo pipefail
cd /workspace-vast/celeste/cnla-skip-lens
export HF_HOME=/workspace-vast/pretrained_ckpts HF_HUB_OFFLINE=1 PYTHONPATH=$PWD WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # cut fragmentation for the big backward
V=/workspace-vast/celeste/oracle-lens/.venv/bin/python
MT=/workspace-vast/celeste/multi-token-jlens-nla-lastlayer
# STANDARD (non-compositional) NLA reconstruction-RL on the NAIVE future-lens with a FROZEN AR.
# Reproduces the original rl_ae_L62 setup EXCEPT: AR frozen (--no-train-critic) + high batch.
STEPS=${STEPS:-200}; BP=${BP:-8}; GS=${GS:-64}; EVERY=${EVERY:-50}; KL=${KL:-0.1}
echo "=== node $(hostname) | naive_rl frozen-AR bp=$BP gs=$GS (=$((BP*GS))/step) kl=$KL ==="
$V -m nla.train_rl_self_contained \
  --base-ckpt Qwen/Qwen3.6-27B --quant none --device-map auto \
  --av-ckpt "$MT/ckpts/av_flv4_L62/iter_0001000" \
  --ar-ckpt /workspace-vast/celeste/easynla-ae/ckpts/ar_L62/iter_0001500 \
  --sidecar "$MT/data/ae_L62/train.parquet" \
  --rl-parquet "$MT/data/ae_L62/train.parquet" \
  --no-train-critic \
  --num-steps "$STEPS" --batch-prompts "$BP" --group-size "$GS" \
  --logp-micro-batch 8 \
  --kl-beta "$KL" --lr 3e-5 --save-every 20 \
  --eval-every "$EVERY" --text-judges-every "$EVERY" --evals base_fve text_judges \
  --save-dir ckpts/naive_rl_frozenAR_L62
echo "NAIVE_RL_DONE"
