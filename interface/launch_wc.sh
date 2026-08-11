#!/bin/bash
# Launch the WeirdChat compare interface on the B300 (gpu0, port 8811) with repeatafterme + presets.
cd /workspace/cnla/skip-lens/interface
source /workspace/.keys.env
export CUDA_VISIBLE_DEVICES=0 PORT=8811
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens  # HF_TOKEN comes from sourced /workspace/.keys.env
export JDIR=/workspace/cnla/results/jlens_official_ws   # OFFICIAL J-lens (camilablank/workspace-lenses), matched with R
export HEADS=/workspace/cnla/results/NONEXISTENT PRESETS=/workspace/cnla/data/weirdchat_rich.json
export AO_CKPT=/workspace/cnla/skip-lens/ckpts/cnla_av_L62/iter_0000300
export NAIVE_CKPT=/workspace/cnla/skip-lens/ckpts/futurelens_scaled_L62/iter_0007579
export RL_CKPT=/workspace/cnla/skip-lens/ckpts/fvecmp_futurelens_b300/iter_000150
export REPEAT_CKPT=/workspace/cnla/adapters/repeatafterme
export REPEAT25K_CKPT=/workspace/cnla/adapters/repeatafterme_25k_allmodules
export REPEAT_SPAN4_CKPT=/workspace/cnla/adapters/repeat_span4_25k_iter1500  # extra "repeat" registry entry (lazy-loaded; skipped if dir missing)
export CNLA_LH_CKPT=/workspace/cnla/skip-lens/ckpts/cnla_longhorizon/iter_000300
export L42M_CKPT=/workspace/cnla/adapters/l42_matched
export L62MM_CKPT=/workspace/cnla/adapters/l62_mismatch
export RDIR=/workspace/cnla/results/rlens_official   # OFFICIAL R-lens (camilablank/workspace-lenses)
export BITTER_CODE=/workspace-vast/celeste/bitter-lens
export BITTER_CKPT=/workspace-vast/celeste/bitter-lens/runs/l42_direct_batch64_1m/best.pt
exec /workspace/cnla_venv/bin/python weirdchat_lens.py
