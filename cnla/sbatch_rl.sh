#!/bin/bash
#SBATCH --job-name=cnla_rl
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --output=/workspace-vast/celeste/cnla-skip-lens/logs/cnla_rl_%j.out
set -uo pipefail
cd /workspace-vast/celeste/cnla-skip-lens
export HF_HOME=/workspace-vast/pretrained_ckpts HF_HUB_OFFLINE=1 PYTHONPATH=$PWD WANDB_MODE=disabled
V=/workspace-vast/celeste/oracle-lens/.venv/bin/python
AV=$(ls -d ckpts/cnla_av_L62/iter_* 2>/dev/null | sort -V | tail -1)
AR=/workspace-vast/celeste/easynla-ae/ckpts/ar_L62/iter_0001500
SIDE=/workspace-vast/celeste/multi-token-jlens-nla-lastlayer/data/ar_L62/train.parquet
echo "=== node $(hostname) | AV=$AV | AR=$AR ==="
# STEPS/BP/GS overridable for smoke-tests (STEPS=2 BP=2 GS=4 sbatch ...)
STEPS=${STEPS:-400}; BP=${BP:-8}; GS=${GS:-16}; EVERY=${EVERY:-100}
EVALS=${EVALS:-base_fve text_judges}   # text_judges (coherence) needs ANTHROPIC_API_KEY
$V -m cnla.train_cnla_rl \
  --base-ckpt Qwen/Qwen3.6-27B --quant none --device-map single \
  --no-train-critic \
  --av-ckpt "$AV" --ar-ckpt "$AR" \
  --sidecar "$SIDE" --whitener data/cnla/whitener.pt \
  --rl-parquet data/cnla/av_train.parquet --save-dir ckpts/cnla_rl_L62 \
  --num-steps "$STEPS" --batch-prompts "$BP" --group-size "$GS" \
  --max-new-tokens 110 --temperature 1.0 \
  --kl-beta 0.1 --lr 3e-5 --save-every 25 \
  --eval-every "$EVERY" --text-judges-every "$EVERY" --evals $EVALS
echo "CNLA_RL_DONE"
