#!/bin/bash
# AR context-prefix determinacy sweep. Builds ctx-augmented datasets {0,32,128,full} then trains
# one AR arm each (1500 steps) with FULL-module LoRA (r64 a16 rsLoRA, --lora-scope all + fp32
# adapters) so capacity is not a confound. FVE-vs-context-length curve settles bug-vs-info-limit.
set -u
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env 2>/dev/null
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
mkdir -p data/ar_ctx logs ckpts/ctx_sweep
SIDE=data/spans_onpolicy/ar_train.parquet   # sidecar is arm-independent (same template/tokens)

echo "=== building ctx datasets ==="
for N in 0 32 128 full; do
  if [ ! -f data/ar_ctx/ctx${N}_train.parquet ]; then
    python pretrain/build_ar_ctx.py --shards-glob "data/spans_onpolicy/shard_*.parquet" \
      --out-prefix data/ar_ctx/ctx${N} --ctx-tokens $N --n-rows 60000 --seed 0
  fi
done
echo CTX_DATA_BUILT

ar_arm(){ # $1 gpu  $2 N
  CUDA_VISIBLE_DEVICES=$1 setsid python -m nla.train_sft --mode ar \
    --base-ckpt Qwen/Qwen3.6-27B --parquet data/ar_ctx/ctx$2_train.parquet --sidecar "$SIDE" \
    --ar-num-layers 63 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
    --lr 1e-4 --batch-size 32 --num-steps 1500 \
    --heldout-parquet data/ar_ctx/ctx$2_heldout.parquet --heldout-rows 1000 --heldout-every 250 \
    --seed 0 --quant none --device-map single --save-every 100000 --strip-final-norm \
    --save-dir ckpts/ctx_sweep/ctx$2 \
    --wandb-project span-sweep --wandb-name ctx$2_full --wandb-tags ctx,determinacy,fulllora \
    > logs/ctx_$2.log 2>&1 < /dev/null &
  echo "ctx=$2 gpu=$1 pid=$!"
}
ar_arm 1 0;    sleep 8
ar_arm 2 32;   sleep 8
ar_arm 3 128;  sleep 8
ar_arm 4 full; sleep 8
echo CTX_SWEEP_LAUNCHED
