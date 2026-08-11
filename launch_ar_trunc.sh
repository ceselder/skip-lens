#!/bin/bash
# Launch a length-matched (trunc) AR scale-up run. Args: $1=gpu $2=save-dir, then extra flags.
cd /workspace/cnla/skip-lens
source /workspace/.keys.env
GPU=$1; SAVE=$2; shift 2
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
D=/workspace/cnla/data/ar_L62_big_trunc/train.parquet
exec /workspace/cnla_venv/bin/python -m nla.train_sft --mode ar --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$D" --sidecar "$D" --ar-num-layers 63 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope attn \
  --lr 3e-5 --gradient-checkpointing --batch-size 32 --num-steps 15157 \
  --heldout-parquet /workspace/cnla/data/ar_ablation/heldout.parquet --heldout-rows 1000 --heldout-every 500 \
  --seed 0 --quant none --device-map single --save-every 3000 \
  --wandb-project ar-scaling --save-dir "$SAVE" "$@"
