#!/bin/bash
# ARM H — K=3, to locate the multi-token budget of this method.
#
# What the evidence now says:
#   armC  K=1, real state trained, Jbar.h42 tested  -> judged 0.561   WORKS
#   armD  K=8, real states trained, Jbar tested     -> CE 0.187 in-distribution,
#                                                      5.99 on the estimates
#                                                      (+5.80 nats: collapse)
# The difference is per-slot estimate quality: cos(Jbar^(d)h42, true h62[p+d])
# is 0.41-0.50 at d=0 but only 0.10-0.16 for every d>=1. So arm D had one usable
# slot and seven noise slots, and was trained to rely on all eight.
#
# Arm H asks where that breaks: K=3 keeps d=0,1,2 — the horizons with the most
# estimate quality left. Prediction, stated before the run: it lands between
# armC and armD, and if only d=0 is truly usable then even K=3 degrades sharply,
# putting the workable budget for this method at ONE token.
set -u
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens

T=/workspace/data/final/pen3_multislot_train.parquet
V=/workspace/data/final/pen3_multislot_val.parquet
N=$(python -c "import pyarrow.parquet as pq; print(max(200, pq.ParquetFile('$T').metadata.num_rows // 64))")
G=${FORCE_GPU:-$(/workspace/gpu_wait.sh 70000)}
echo "[armH] K=3 training on gpu $G for $N steps"
CUDA_VISIBLE_DEVICES=$G python -m nla.train_sft --mode av \
  --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$T" --sidecar "$T" --n-slots 3 \
  --save-dir ckpts/multislot_armH_pen3 \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$N" --batch-size 16 --gradient-accumulation-steps 4 \
  --save-every 500 --heldout-parquet "$V" --heldout-rows 800 --heldout-every 250 \
  --sample-every 500 --n-samples 4 --sample-max-new-tokens 48 \
  --wandb-project skiplens-multislot --wandb-name armH_pen3_k3
echo "done."
