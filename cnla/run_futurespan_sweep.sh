#!/bin/bash
# Future-span-length sweep: does the AR reconstruct the L62 activation better with MORE future
# tokens? Train an AR per FIXED future-span-length L in {4,8,12,16} (span only, no context),
# 1500 steps each, read heldout FVE. FVE-vs-L answers "does sampling more future help?".
# (Rollouts are 16 tok, so 16 is the max here; extend with a longer re-collection if still rising.)
set -u
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env 2>/dev/null
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
mkdir -p data/ar_ctx logs ckpts/fspan_sweep
SIDE=data/spans_onpolicy/ar_train.parquet

echo "=== building fixed-length future-span datasets ==="
for L in 4 8 12 16; do
  if [ ! -f data/ar_ctx/span${L}_train.parquet ]; then
    python pretrain/build_ar_ctx.py --shards-glob "data/spans_onpolicy/shard_*.parquet" \
      --out-prefix data/ar_ctx/span${L} --ctx-tokens 0 --span-min $L --span-max $L \
      --n-rows 60000 --seed 0
  fi
done
echo FSPAN_DATA_BUILT

ar_arm(){ # $1 gpu  $2 L
  CUDA_VISIBLE_DEVICES=$1 setsid python -m nla.train_sft --mode ar \
    --base-ckpt Qwen/Qwen3.6-27B --parquet data/ar_ctx/span$2_train.parquet --sidecar "$SIDE" \
    --ar-num-layers 63 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
    --lr 1e-4 --batch-size 32 --num-steps 1500 \
    --heldout-parquet data/ar_ctx/span$2_heldout.parquet --heldout-rows 1000 --heldout-every 250 \
    --seed 0 --quant none --device-map single --save-every 100000 --strip-final-norm \
    --save-dir ckpts/fspan_sweep/span$2 \
    --wandb-project span-sweep --wandb-name fspan$2 --wandb-tags fspan,determinacy \
    > logs/fspan_$2.log 2>&1 < /dev/null &
  echo "fspan L=$2 gpu=$1 pid=$!"
}
ar_arm 1 4;  sleep 8
ar_arm 2 8;  sleep 8
ar_arm 3 12; sleep 8
ar_arm 4 16; sleep 8
echo FSPAN_SWEEP_LAUNCHED
