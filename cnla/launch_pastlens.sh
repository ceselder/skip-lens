#!/bin/bash
# Train the PASTLENS skip-lens (activation -> the text that PRECEDED position p) with the same
# AV recipe as the futurelens, on gpu0 (co-located with the interface; 275GB GPU has room).
set -u
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env 2>/dev/null
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
mkdir -p logs ckpts
PL_TRAIN=data/spans_onpolicy/pastlens_av_train.parquet
ROWS=$(python -c "import pyarrow.parquet as pq;print(pq.ParquetFile('$PL_TRAIN').metadata.num_rows)")
STEPS=$((ROWS/64))
echo "PASTLENS rows=$ROWS steps=$STEPS (1 epoch, eff batch 64)"
CUDA_VISIBLE_DEVICES=0 setsid python -m nla.train_sft --mode av \
  --base-ckpt Qwen/Qwen3.6-27B --parquet "$PL_TRAIN" --sidecar "$PL_TRAIN" \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --lr 1e-4 --num-steps $STEPS --batch-size 16 --gradient-accumulation-steps 4 \
  --seed 0 --quant none --device-map single --save-every 3000 \
  --save-dir ckpts/pastlens_L62 \
  --wandb-project span-sweep --wandb-name pastlens_L62 --wandb-tags span,pastlens,lens \
  > logs/pastlens_train.log 2>&1 < /dev/null &
echo "PASTLENS_TRAIN pid=$!"
