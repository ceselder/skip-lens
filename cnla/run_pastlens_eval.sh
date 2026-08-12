#!/bin/bash
# Quick pastlens-vs-futurelens read: inject held-out activations, generate, val-CE + samples.
# pastlens reconstructs the PAST context; futurelens (same activations) predicts the FUTURE span.
set -u
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env 2>/dev/null
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
PL=$(ls -d ckpts/pastlens_L62/iter_* | sort -V | tail -1)
FL=$(ls -d ckpts/span_sweep/av_lr3e-5/iter_* | sort -V | tail -1)
echo "pastlens=$PL futurelens=$FL"
CUDA_VISIBLE_DEVICES=1 setsid python -m evals.av_valce_diag \
  --av-ckpt "$PL" --ctx-parquet data/spans_onpolicy/pastlens_av_val.parquet --n 64 --tag pastlens \
  > logs/eval_pastlens.log 2>&1 < /dev/null &
echo "pastlens_eval pid=$!"
CUDA_VISIBLE_DEVICES=3 setsid python -m evals.av_valce_diag \
  --av-ckpt "$FL" --ctx-parquet data/spans_onpolicy/av_val.parquet --n 64 --tag flens3e5 \
  > logs/eval_flens.log 2>&1 < /dev/null &
echo "flens_eval pid=$!"
