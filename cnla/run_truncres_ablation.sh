#!/bin/bash
# Truncation-resistance @ span16: dense --ar-all-idx (reconstruct at EVERY causal prefix) vs
# normal last-token AR. Both log FVE by distance-from-last-token (--ar-fve-by-dist).
set -u
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate; source /workspace/.keys.env 2>/dev/null
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
mkdir -p logs ckpts/truncres
C="--base-ckpt Qwen/Qwen3.6-27B --parquet data/ar_ctx/span16_train.parquet --sidecar data/spans_onpolicy/ar_train.parquet --ar-num-layers 63 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all --lr 1e-4 --batch-size 32 --num-steps 1500 --heldout-parquet data/ar_ctx/span16_heldout.parquet --heldout-rows 1000 --heldout-every 250 --seed 0 --quant none --device-map single --save-every 100000 --strip-final-norm --ar-fve-by-dist"
CUDA_VISIBLE_DEVICES=1 setsid python -m nla.train_sft --mode ar $C \
  --save-dir ckpts/truncres/lasttok --wandb-project span-sweep --wandb-name truncres_last --wandb-tags truncres,last \
  > logs/truncres_last.log 2>&1 < /dev/null &
echo "last-token arm pid=$!"; sleep 8
CUDA_VISIBLE_DEVICES=2 setsid python -m nla.train_sft --mode ar $C --ar-all-idx \
  --save-dir ckpts/truncres/allidx --wandb-project span-sweep --wandb-name truncres_allidx --wandb-tags truncres,allidx \
  > logs/truncres_allidx.log 2>&1 < /dev/null &
echo "all-idx arm pid=$!"; echo TRUNCRES_LAUNCHED
