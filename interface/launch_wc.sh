#!/bin/bash
# Launch the WeirdChat compare interface on the B300 (gpu0, port 8811) with repeatafterme + presets.
cd /workspace/cnla/skip-lens/interface
source /workspace/.keys.env
export CUDA_VISIBLE_DEVICES=0 PORT=8811
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens  # HF_TOKEN comes from sourced /workspace/.keys.env
export JDIR=/workspace/cnla/results/jlens_snap
export HEADS=/workspace/cnla/results/NONEXISTENT PRESETS=/workspace/cnla/data/weirdchat_rich.json
export AO_CKPT=/workspace/cnla/skip-lens/ckpts/cnla_av_L62/iter_0000300
export NAIVE_CKPT=/workspace/cnla/skip-lens/ckpts/futurelens_scaled_L62/iter_0007579
export RL_CKPT=/workspace/cnla/skip-lens/ckpts/fvecmp_futurelens_b300/iter_000150
export REPEAT_CKPT=/workspace/cnla/adapters/repeatafterme
export REPEAT25K_CKPT=/workspace/cnla/adapters/repeatafterme_25k_allmodules
export CNLA_LH_CKPT=/workspace/cnla/skip-lens/ckpts/cnla_longhorizon/iter_000300
exec /workspace/cnla_venv/bin/python weirdchat_lens.py
