#!/bin/bash
#SBATCH --job-name=bitter_pg
#SBATCH --qos=high
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace-vast/celeste/cnla-skip-lens/logs/bitter_playground.log

set -euo pipefail

# Importing the large shared venv can block for minutes in NFS RPC waits. The
# compute image already has CUDA PyTorch, so keep the thin Python layer local.
LOCAL_ENV=/tmp/celeste-bitter-playground-env
if [[ ! -x "$LOCAL_ENV/bin/python" ]]; then
  python3 -m venv --system-site-packages "$LOCAL_ENV"
  "$LOCAL_ENV/bin/pip" install --disable-pip-version-check --no-cache-dir \
    transformers==5.5.4 peft==0.19.1 fastapi==0.139.2 uvicorn==0.51.0
fi
if ! "$LOCAL_ENV/bin/python" -c 'import pyarrow' >/dev/null 2>&1; then
  "$LOCAL_ENV/bin/pip" install --disable-pip-version-check --no-cache-dir pyarrow
fi
source "$LOCAL_ENV/bin/activate"

ROOT=/workspace-vast/celeste/cnla-skip-lens
export HF_HOME=/workspace-vast/pretrained_ckpts
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT"
export PORT=8811

export BASE_CKPT=Qwen/Qwen3.6-27B
export TARGET_L=62
export JDIR=/workspace-vast/celeste/multi-token-jlens-nla-lastlayer/results/jlens_official
export HEADS=/workspace-vast/celeste/cnla-skip-lens/results/no_horizon_heads
export PRESETS=/workspace-vast/celeste/cnla-skip-lens/data/no_presets.json

# Skip-Lens verbalizer and the activation-conditioned direct Bitter checkpoint.
export AO_CKPT="$ROOT/ckpts/cnla_av_L62/iter_0000300"
export NAIVE_CKPT="$ROOT/ckpts/futurelens_big_L62/iter_0002000"
export BITTER_CODE=/workspace-vast/celeste/bitter-lens
export BITTER_CKPT=/workspace-vast/celeste/bitter-lens/runs/l42_direct_batch64_1m/best.pt

cd "$ROOT/interface"
python weirdchat_lens.py
