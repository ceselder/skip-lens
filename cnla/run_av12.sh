#!/bin/bash
# Standalone AV (NLA/futurelens) SFT, no co-training. On-policy spans capped at 12 tokens.
# Effective batch 2048 (micro 16 x grad-accum 128) for training stability. Full LoRA r64 a16 rsLoRA.
# Trains on the SEEN spans_onpolicy pool (RL will use a fresh disjoint pool).
set -u
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env 2>/dev/null
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
GPU=${GPU:-1}
mkdir -p logs ckpts

# 1) AV data, on-policy spans capped at <=12 tokens
if [ ! -f data/spans_onpolicy/av12_train.parquet ]; then
  echo "[av12] building span<=12 data..."
  python pretrain/finalize_span_data.py \
    --shards-glob "data/spans_onpolicy/shard_*.parquet" \
    --meta data/spans_onpolicy/shard_0.parquet.meta.json \
    --out-train data/spans_onpolicy/av12_train.parquet \
    --out-val   data/spans_onpolicy/av12_val.parquet \
    --span-source rollout --span-min 4 --span-max 12 --seed 0
fi
ROWS=$(python -c "import pyarrow.parquet as pq;print(pq.ParquetFile('data/spans_onpolicy/av12_train.parquet').metadata.num_rows)")
STEPS=$((ROWS/2048))
echo "[av12] rows=$ROWS eff_batch=2048 (64x32) steps=$STEPS gpu=$GPU"

CUDA_VISIBLE_DEVICES=$GPU setsid python -m nla.train_sft --mode av \
  --base-ckpt Qwen/Qwen3.6-27B --parquet data/spans_onpolicy/av12_train.parquet \
  --sidecar data/spans_onpolicy/av12_train.parquet \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --lr 1e-4 --batch-size 64 --gradient-accumulation-steps 32 --num-steps $STEPS \
  --seed 0 --quant none --device-map single --save-every 100 \
  --save-dir ckpts/av12_L62 \
  --wandb-project span-sweep --wandb-name av12_bs2048 --wandb-tags av,span12,bs2048,stable \
  > logs/av12_train.log 2>&1 < /dev/null &
echo "AV12_TRAIN pid=$!"
