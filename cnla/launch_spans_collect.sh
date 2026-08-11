#!/bin/bash
# ON-POLICY collection across gpu1-7: each row = L62 residual at pos p + a span the
# MODEL ACTUALLY GENERATES from the doc prefix (model.generate, sampled temp=1.0). The
# AR is a reward for on-policy RL, so its spans must be the model's own generations, NOT
# corpus text. rollout-len 16 -> finalize slices rollout_token_ids[0][:k], k~U[4,16].
# 100k FineFineWeb docs sharded 7 ways, 8 positions/doc -> ~0.8M on-policy pairs.
set -u
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env 2>/dev/null
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
mkdir -p data/spans_onpolicy logs
CORPUS=/workspace/cnla/corpus/finefineweb_100k.parquet
NDOCS=14286   # 100000 / 7 (last shard clamps at corpus end)
for i in 0 1 2 3 4 5 6; do
  GPU=$((i+1))
  CUDA_VISIBLE_DEVICES=$GPU setsid python pretrain/collect_ao_data.py \
    --base-ckpt Qwen/Qwen3.6-27B \
    --corpus "$CORPUS" \
    --out data/spans_onpolicy/shard_${i}.parquet \
    --layer 62 --n-docs $NDOCS --doc-offset $((i*NDOCS)) \
    --positions-per-doc 8 --rollouts 1 --rollout-len 16 --seed $i \
    > logs/onpolicy_collect_${i}.log 2>&1 < /dev/null &
  echo "launched shard $i on gpu $GPU pid $!"
  sleep 12   # stagger so the first model load warms the page cache before the rest
done
echo "all 7 on-policy collection shards launched"
