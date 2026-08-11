#!/bin/bash
#SBATCH --job-name=fvecmp_fl
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=/workspace-vast/celeste/cnla-skip-lens/logs/fvecmp_fl_%j.out
set -uo pipefail
cd /workspace-vast/celeste/cnla-skip-lens
export HF_HOME=/workspace-vast/pretrained_ckpts HF_HUB_OFFLINE=1 PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /workspace-vast/celeste/.keys.env
V=/workspace-vast/celeste/oracle-lens/.venv/bin/python
# === FUTURELENS ARM of the data-matched FVE comparison ===
# naive future-lens (single continuation) init, SFT'd on the SAME collect_L62 data as the CNLA AV.
# Standard whole-readout reconstruction RL with a FROZEN AR, on the SAME activation set (av_train).
AV=$(ls -d ckpts/naive_av_matched_L62/iter_* 2>/dev/null | sort -V | tail -1)
AR=/workspace-vast/celeste/easynla-ae/ckpts/ar_L62/iter_0001500
SIDE=/workspace-vast/celeste/multi-token-jlens-nla-lastlayer/data/ar_L62/train.parquet
STEPS=${STEPS:-150}; BP=${BP:-8}; GS=${GS:-128}; EVERY=${EVERY:-50}; KL=${KL:-0.1}
echo "=== node $(hostname) | FUTURELENS ARM | AV=$AV | bp=$BP gs=$GS (=$((BP*GS))/step) ==="
$V -m nla.train_rl_self_contained \
  --base-ckpt Qwen/Qwen3.6-27B --quant none --device-map auto \
  --av-ckpt "$AV" --ar-ckpt "$AR" \
  --sidecar "$SIDE" --rl-parquet data/cnla/av_train.parquet \
  --no-train-critic \
  --num-steps "$STEPS" --batch-prompts "$BP" --group-size "$GS" --logp-micro-batch 2 \
  --kl-beta "$KL" --lr 3e-5 --save-every 25 \
  --eval-every "$EVERY" --text-judges-every "$EVERY" --evals base_fve text_judges \
  --wandb-project cnla-fve-compare --wandb-group fve-compare-datamatched \
  --wandb-name futurelens_arm --wandb-tags data-matched,frozen-ar,8x128 \
  --save-dir ckpts/fvecmp_futurelens_L62
echo "FVECMP_FL_DONE"
