#!/bin/bash
# Jacobian-arm fed-layer sweep for ONE skip-lens checkpoint, then Sonnet judge.
# Args: $1=gpu $2=ao-ckpt $3=tag. Feeds J_{l->62}.h_l (fitted Jacobian into penultimate basis).
set -uo pipefail
cd /workspace/cnla/skip-lens/evals
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
export CUDA_VISIBLE_DEVICES=$1
CKPT=$2; TAG=$3
OUT=/workspace/cnla/results/layerabl; mkdir -p "$OUT"
echo "=== fedlayer JAC sweep: $TAG | $CKPT | gpu $1 ==="
python fedlayer_jac_eval.py --ao-ckpt "$CKPT" --jdir /workspace/cnla/results/jlens_snap \
  --jorigin 42 --jtarget 62 --fed-layers "55,48,42,34,26,18,10" \
  --evals-dir /workspace/cnla/skip-lens/evals/datasets_fed \
  --out "$OUT/fedjac_${TAG}.json"
echo "=== judge (Sonnet-5): $TAG jac ==="
python judge_fedlayer.py --in "$OUT/fedjac_${TAG}.json" --out "$OUT/judgedjac_${TAG}.json" --model claude-sonnet-5
echo "JACPIPELINE_DONE_${TAG}"
