#!/bin/bash
# Data-scaling futurelens pretraining: train ONE epoch (no repeats) over the big SFT set,
# checkpoint every 10th of the run -> each checkpoint = "trained on N distinct examples" =
# the x-axis of the "does more DATA help skip-lens" curve. Run on B300 (GPU env var).
set -uo pipefail
cd /workspace/cnla/skip-lens && source /workspace/cnla_venv/bin/activate && source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
export CUDA_VISIBLE_DEVICES=${GPU:-2}
ROWS=$(python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('data/fl_big/sft_train.parquet').metadata.num_rows)")
NSTEPS=$(( (ROWS + 63) / 64 ))          # 1 epoch @ batch 16 x gradaccum 4 = 64/step
SAVE=$(( NSTEPS / 10 )); [ $SAVE -lt 1 ] && SAVE=1
echo "=== scaled futurelens | $ROWS rows -> $NSTEPS steps (1 epoch) | save every $SAVE | gpu $CUDA_VISIBLE_DEVICES ==="
python -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
  --parquet data/fl_big/sft_train.parquet --sidecar data/fl_big/sft_train.parquet \
  --save-dir ckpts/futurelens_scaled_L62 \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$NSTEPS" --batch-size 16 --gradient-accumulation-steps 4 --save-every "$SAVE" \
  --wandb-project futurelens-pretrain-scaling --wandb-group data-scaling --wandb-name futurelens_scaled \
  --wandb-tags data-scaling,1epoch
echo SCALED_TRAIN_DONE
