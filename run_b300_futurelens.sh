#!/bin/bash
# FUTURELENS ARM on B300 (no SLURM). One B300 (275GB) holds actor+AR (~108GB) + a big backward.
set -uo pipefail
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
export CUDA_VISIBLE_DEVICES=${GPU:-0}
source /workspace/.keys.env
AV=${AVDIR:-$(ls -d ckpts/naive_av_matched_L62/iter_* 2>/dev/null | sort -V | tail -1)}
AR=${AR:-/workspace/cnla/ckpts/ar_L62/iter_0001500}
SIDE=${SIDE:-/workspace/cnla/data/sidecar_ar_L62_train.parquet}
STEPS=${STEPS:-150}; BP=${BP:-8}; GS=${GS:-128}; EVERY=${EVERY:-50}; KL=${KL:-0.1}; MB=${MB:-16}
RLP=${RLP:-data/cnla/av_train.parquet}; SAVEDIR=${SAVEDIR:-ckpts/fvecmp_futurelens_b300}
WGROUP=${WGROUP:-fve-compare-b300}; WNAME=${WNAME:-futurelens_arm_b300}
FEEDARG=${FEEDCOL:+--feed-col $FEEDCOL}   # feed a different layer (e.g. act_L42) than the L62 gold
echo "=== B300 FUTURELENS | gpu=$CUDA_VISIBLE_DEVICES | AV=$AV | rlp=$RLP | bp=$BP gs=$GS mb=$MB ==="
python -m nla.train_rl_self_contained \
  --base-ckpt Qwen/Qwen3.6-27B --quant none --device-map single \
  --av-ckpt "$AV" --ar-ckpt "$AR" \
  --sidecar "$SIDE" --rl-parquet "$RLP" $FEEDARG \
  --no-train-critic \
  --num-steps "$STEPS" --batch-prompts "$BP" --group-size "$GS" --logp-micro-batch "$MB" \
  --kl-beta "$KL" --lr 3e-5 --save-every 25 \
  --eval-every "$EVERY" --text-judges-every "$EVERY" --evals base_fve text_judges \
  --wandb-project cnla-fve-compare --wandb-group "$WGROUP" \
  --wandb-name "$WNAME" --wandb-tags fedlayer,b300 \
  --save-dir "$SAVEDIR"
echo "B300_FL_DONE"
