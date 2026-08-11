#!/bin/bash
# Launch one arm of the source-layer ablation AV-SFT. Args: $1=gpu $2=data.parquet $3=save-dir-name $4=num-steps $5=save-every
# Recipe is verbatim run_scaled_train.sh (the canonical futurelens skip-lens recipe) so the only
# cross-arm difference is the source layer baked into the data's activation_vector.
set -uo pipefail
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
export CUDA_VISIBLE_DEVICES=$1
DATA=$2; NAME=$3; NSTEPS=$4; SAVE=$5
echo "=== layer-ablation AV-SFT | gpu=$1 | data=$DATA | save-dir=ckpts/$NAME | steps=$NSTEPS save-every=$SAVE ==="
exec python -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$DATA" --sidecar "$DATA" \
  --save-dir "ckpts/$NAME" \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$NSTEPS" --batch-size 16 --gradient-accumulation-steps 4 --save-every "$SAVE" \
  --wandb-project skiplens-layer-ablation --wandb-group source-layer --wandb-name "$NAME" \
  --wandb-tags layer-ablation,150k
