#!/bin/bash
# LR sweep on the REAL-span data across gpu1-7 (gpu0 = interface).
#   AR (span->activation, <summary> anchor, MSE/FVE): LR in {1e-5, 3e-5, 1e-4, 3e-4}
#   AV/futurelens (activation->span, CE, inverse task, SAME data): LR in {3e-5, 1e-4, 3e-4}
# Each run does 1 epoch on the big set; heldout FVE every 500 steps gives the LR ranking early.
set -u
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env 2>/dev/null
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
mkdir -p logs ckpts/span_sweep

AR_TRAIN=data/spans_onpolicy/ar_train.parquet
AR_HELD=data/spans_onpolicy/av_val.parquet   # heldout loader needs AV-format (response+activation), NOT ar_heldout
AV_TRAIN=data/spans_onpolicy/av_train.parquet
ARM=${ARM:-all}   # all | ar | av  (relaunch a single arm without touching the other)

AR_ROWS=$(python -c "import pyarrow.parquet as pq;print(pq.ParquetFile('$AR_TRAIN').metadata.num_rows)")
AV_ROWS=$(python -c "import pyarrow.parquet as pq;print(pq.ParquetFile('$AV_TRAIN').metadata.num_rows)")
AR_STEPS=$((AR_ROWS/32))         # batch 32, 1 epoch
AV_STEPS=$((AV_ROWS/64))         # eff batch 16*4, 1 epoch
echo "AR_ROWS=$AR_ROWS AR_STEPS=$AR_STEPS ; AV_ROWS=$AV_ROWS AV_STEPS=$AV_STEPS"

ar_run(){  # $1=gpu $2=lr
  CUDA_VISIBLE_DEVICES=$1 setsid python -m nla.train_sft --mode ar \
    --base-ckpt Qwen/Qwen3.6-27B --parquet "$AR_TRAIN" --sidecar "$AR_TRAIN" \
    --ar-num-layers 63 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope attn \
    --lr $2 --gradient-checkpointing --batch-size 32 --num-steps $AR_STEPS \
    --heldout-parquet "$AR_HELD" --heldout-rows 1000 --heldout-every 500 \
    --seed 0 --quant none --device-map single --save-every 5000 --strip-final-norm \
    --save-dir ckpts/span_sweep/ar_lr$2 \
    --wandb-project span-sweep --wandb-name ar_lr$2 --wandb-tags span,ar,lrsweep \
    > logs/sweep_ar_lr$2.log 2>&1 < /dev/null &
  echo "AR lr=$2 gpu=$1 pid=$!"
}
av_run(){  # $1=gpu $2=lr
  CUDA_VISIBLE_DEVICES=$1 setsid python -m nla.train_sft --mode av \
    --base-ckpt Qwen/Qwen3.6-27B --parquet "$AV_TRAIN" --sidecar "$AV_TRAIN" \
    --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
    --lr $2 --num-steps $AV_STEPS --batch-size 16 --gradient-accumulation-steps 4 \
    --seed 0 --quant none --device-map single --save-every 3000 \
    --save-dir ckpts/span_sweep/av_lr$2 \
    --wandb-project span-sweep --wandb-name av_lr$2 --wandb-tags span,av,lrsweep \
    > logs/sweep_av_lr$2.log 2>&1 < /dev/null &
  echo "AV lr=$2 gpu=$1 pid=$!"
}

if [ "$ARM" != av ]; then
  ar_run 1 1e-5; sleep 10
  ar_run 2 3e-5; sleep 10
  ar_run 3 1e-4; sleep 10
  ar_run 4 3e-4; sleep 10
fi
if [ "$ARM" != ar ]; then
  av_run 5 3e-5; sleep 10
  av_run 6 1e-4; sleep 10
  av_run 7 3e-4; sleep 10
fi
echo "SWEEP_LAUNCHED (arm=$ARM): AR{1e-5,3e-5,1e-4,3e-4} gpu1-4 ; AV{3e-5,1e-4,3e-4} gpu5-7"
