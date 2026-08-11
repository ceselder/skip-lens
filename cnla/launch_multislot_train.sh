#!/bin/bash
# Data prep + training for the multi-slot experiment. Two arms:
#   armA (gpu $1): K=8 slot AV on LOCAL-JVP transported spans (--n-slots 8)
#   armC (gpu $2): single-slot AV on raw L62 activations, SAME docs, 8-token
#                  spans (token-matched baseline; test-time gets pooled Jbar.h42)
# Recipe otherwise verbatim run_scaled_train.sh / launch_layerabl.sh
# (LoRA r64 a16 rsLoRA scope=all, eff batch 64, default lr).
# Usage: bash cnla/launch_multislot_train.sh 0 1
set -uo pipefail
GPU_A=${1:-0}; GPU_C=${2:-1}
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens
mkdir -p logs ckpts /workspace/data/final

AV_A=/workspace/data/final/jvp_multislot_train.parquet
AV_A_VAL=/workspace/data/final/jvp_multislot_val.parquet
AV_C=/workspace/data/final/spanC_L62_train.parquet
AV_C_VAL=/workspace/data/final/spanC_L62_val.parquet

if [ ! -f "$AV_A" ]; then
  echo "=== finalize arm A (multi-slot JVP spans, K=8) ==="
  python pretrain/finalize_jvp_spans.py \
    --shards-glob "/workspace/data/spans_jvp/shard_*_jvp.parquet" \
    --out-train "$AV_A" --out-val "$AV_A_VAL" --k-slots 8
fi
if [ ! -f "$AV_C" ]; then
  echo "=== finalize arm C (single-slot L62, 8-token spans) ==="
  python pretrain/finalize_span_data.py \
    --shards-glob "/workspace/data/spans_raw/shard_*.parquet" \
    --meta /workspace/data/spans_raw/shard_0.parquet.meta.json \
    --out-train "$AV_C" --out-val "$AV_C_VAL" \
    --span-min 8 --span-max 8 --span-source rollout
fi

steps_for(){ python -c "import pyarrow.parquet as pq;print(max(200,pq.ParquetFile('$1').metadata.num_rows//64))"; }
NA=$(steps_for "$AV_A"); NC=$(steps_for "$AV_C")
echo "armA steps=$NA armC steps=$NC"

CUDA_VISIBLE_DEVICES=$GPU_A setsid python -m nla.train_sft --mode av \
  --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$AV_A" --sidecar "$AV_A" --n-slots 8 \
  --save-dir ckpts/multislot_armA_k8 \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$NA" --batch-size 16 --gradient-accumulation-steps 4 \
  --save-every 500 --heldout-parquet "$AV_A_VAL" --heldout-rows 800 --heldout-every 250 \
  --sample-every 500 --n-samples 4 --sample-max-new-tokens 48 \
  --wandb-project skiplens-multislot --wandb-name armA_jvp_k8 \
  > logs/train_armA.log 2>&1 < /dev/null &
echo "armA on gpu $GPU_A pid $!"

CUDA_VISIBLE_DEVICES=$GPU_C setsid python -m nla.train_sft --mode av \
  --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$AV_C" --sidecar "$AV_C" \
  --save-dir ckpts/multislot_armC_L62 \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$NC" --batch-size 16 --gradient-accumulation-steps 4 \
  --save-every 500 --heldout-parquet "$AV_C_VAL" --heldout-rows 800 --heldout-every 250 \
  --sample-every 500 --n-samples 4 --sample-max-new-tokens 48 \
  --wandb-project skiplens-multislot --wandb-name armC_L62_span8 \
  > logs/train_armC.log 2>&1 < /dev/null &
echo "armC on gpu $GPU_C pid $!"
echo "both arms launched"
