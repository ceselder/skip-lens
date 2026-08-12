#!/bin/bash
# Launch the multi-slot workspace-lens playground (arm A conditions + arm C
# baseline). Standalone copy — does NOT use or modify interface/ (sibling
# session's working dir), only imports its common.py read-only.
# Usage: bash interface_multislot/launch_multislot_pg.sh [gpu] [port]
set -u
GPU=${1:-1}
PORT=${2:-8807}
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens:/workspace/skip-lens/interface
export CUDA_VISIBLE_DEVICES=$GPU PORT=$PORT
export JBAR_DIR=/workspace/results/offset_jlens
export ARMA_CKPT=/workspace/skip-lens/ckpts/multislot_armA_k8/iter_0003875
export ARMC_CKPT=/workspace/skip-lens/ckpts/multislot_armC_L62/iter_0003876
exec python interface_multislot/multislot_playground.py
