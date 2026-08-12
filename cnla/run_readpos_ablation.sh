#!/bin/bash
# Read-position ablation at FIXED span length 16: does last-token vs a dedicated <|summary|>
# read-anchor still matter? (Earlier finding on variable/short spans favored the anchor.)
# Both arms: AR reconstruct L62 from a 16-token on-policy span, lr 1e-4, 1500 steps, identical
# except the read position. FVE-vs-read-position is the answer.
set -u
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env 2>/dev/null
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
mkdir -p data/ar_ctx logs ckpts/readpos

# span16 (last-token) data already exists from the future-span sweep; build the sumtok variant
if [ ! -f data/ar_ctx/span16sum_train.parquet ]; then
  python pretrain/build_ar_ctx.py --shards-glob "data/spans_onpolicy/shard_*.parquet" \
    --out-prefix data/ar_ctx/span16sum --ctx-tokens 0 --span-min 16 --span-max 16 --summary-token \
    --n-rows 60000 --seed 0
fi
echo READPOS_DATA_READY

# arm A: LAST-TOKEN read (standard "</text> <summary>" suffix)
CUDA_VISIBLE_DEVICES=1 setsid python -m nla.train_sft --mode ar \
  --base-ckpt Qwen/Qwen3.6-27B --parquet data/ar_ctx/span16_train.parquet \
  --sidecar data/spans_onpolicy/ar_train.parquet \
  --ar-num-layers 63 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --lr 1e-4 --batch-size 32 --num-steps 1500 \
  --heldout-parquet data/ar_ctx/span16_heldout.parquet --heldout-rows 1000 --heldout-every 250 \
  --seed 0 --quant none --device-map single --save-every 100000 --strip-final-norm \
  --save-dir ckpts/readpos/span16_last \
  --wandb-project span-sweep --wandb-name readpos_last16 --wandb-tags readpos,last,span16 \
  > logs/readpos_last16.log 2>&1 < /dev/null &
echo "LAST-token arm pid=$!"; sleep 8

# arm B: dedicated <|summary|> read anchor (--ar-summary-token; data ends with <|summary|>)
CUDA_VISIBLE_DEVICES=2 setsid python -m nla.train_sft --mode ar \
  --base-ckpt Qwen/Qwen3.6-27B --parquet data/ar_ctx/span16sum_train.parquet \
  --sidecar /workspace/cnla/data/ar_ablation/sumtok_train.parquet \
  --ar-num-layers 63 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all --ar-summary-token \
  --lr 1e-4 --batch-size 32 --num-steps 1500 \
  --heldout-parquet data/ar_ctx/span16sum_heldout.parquet --heldout-rows 1000 --heldout-every 250 \
  --seed 0 --quant none --device-map single --save-every 100000 --strip-final-norm \
  --save-dir ckpts/readpos/span16_sum \
  --wandb-project span-sweep --wandb-name readpos_sum16 --wandb-tags readpos,sumtok,span16 \
  > logs/readpos_sum16.log 2>&1 < /dev/null &
echo "SUMMARY-token arm pid=$!"
echo READPOS_LAUNCHED
