#!/bin/bash
# Workspace-agreement fed-layer sweep for ONE skip-lens checkpoint, then Sonnet judge.
# Args: $1=gpu $2=ao-ckpt $3=tag. Feeds RAW residual at each depth (the workspace condition);
# judge scores agree_jlens (workspace) vs agree_answer (surface) per fed-layer.
set -uo pipefail
cd /workspace/cnla/skip-lens/evals
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
export CUDA_VISIBLE_DEVICES=$1
CKPT=$2; TAG=$3
OUT=/workspace/cnla/results/layerabl; mkdir -p "$OUT"
echo "=== fedlayer sweep (raw h_l): $TAG | $CKPT | gpu $1 ==="
python fedlayer_sweep_eval.py --ao-ckpt "$CKPT" --jdir /workspace/cnla/results/jlens_snap \
  --jorigin 42 --jtarget 62 --fed-layers "62,55,48,42,34,26,18,10" \
  --evals-dir /workspace/cnla/skip-lens/evals/datasets_fed \
  --out "$OUT/fed_${TAG}.json"
echo "=== judge (Sonnet-5): $TAG ==="
python judge_fedlayer.py --in "$OUT/fed_${TAG}.json" --out "$OUT/judged_${TAG}.json" --model claude-sonnet-5
echo "PIPELINE_DONE_${TAG}"
