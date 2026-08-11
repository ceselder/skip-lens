#!/bin/bash
# CNLA (BULLETS) ARM on B300 (no SLURM). One B300 (275GB) holds actor+AR (~108GB) + a big backward.
set -uo pipefail
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
export CUDA_VISIBLE_DEVICES=${GPU:-1}
source /workspace/.keys.env
AV=$(ls -d ckpts/cnla_av_L62/iter_* 2>/dev/null | sort -V | tail -1)
AR=/workspace/cnla/ckpts/ar_L62/iter_0001500
SIDE=/workspace/cnla/data/sidecar_ar_L62_train.parquet
STEPS=${STEPS:-150}; BP=${BP:-8}; GS=${GS:-128}; EVERY=${EVERY:-50}; KL=${KL:-0.1}; MB=${MB:-16}
RLP=${RLP:-data/cnla/av_train.parquet}; SAVEDIR=${SAVEDIR:-ckpts/fvecmp_cnla_b300}
WGROUP=${WGROUP:-fve-compare-b300}; WNAME=${WNAME:-cnla_arm_b300}
FEEDARG=${FEEDCOL:+--feed-col $FEEDCOL}   # feed a different layer (e.g. act_L42) than the L62 gold
LOOARG=""; [ -n "${LOOTHR:-}" ] && LOOARG="$LOOARG --loo-threshold $LOOTHR"
[ -n "${MISSPEN:-}" ] && LOOARG="$LOOARG --missing-bullet-penalty $MISSPEN"   # NEGATIVE to penalize truncation
echo "=== B300 CNLA | gpu=$CUDA_VISIBLE_DEVICES | AV=$AV | rlp=$RLP | bp=$BP gs=$GS mb=$MB ==="
python -m cnla.train_cnla_rl \
  --base-ckpt Qwen/Qwen3.6-27B --quant none --device-map single \
  --av-ckpt "$AV" --ar-ckpt "$AR" \
  --sidecar "$SIDE" --whitener data/cnla/whitener.pt \
  --rl-parquet "$RLP" $FEEDARG $LOOARG \
  --no-train-critic \
  --num-steps "$STEPS" --batch-prompts "$BP" --group-size "$GS" --logp-micro-batch "$MB" \
  --max-new-tokens 110 --temperature 1.0 \
  --kl-beta "$KL" --lr 3e-5 --save-every 25 \
  --eval-every "$EVERY" --text-judges-every "$EVERY" --evals base_fve text_judges \
  --wandb-project cnla-fve-compare --wandb-group "$WGROUP" \
  --wandb-name "$WNAME" --wandb-tags fedlayer,b300 \
  --save-dir "$SAVEDIR"
echo "B300_CNLA_DONE"
