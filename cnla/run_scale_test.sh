#!/bin/bash
# Scale test: does the mean-centering +0.119 survive at 10x data, or was it just rescuing an
# undertrained 500-pair lens? Train raw_5k + meansub_norm_5k (same recipe, 500 steps), eval L62+L42,
# judge. Same raw substrate + own-layer local-J 8-tok scoring as the 500-pair comparison.
set -uo pipefail
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
ROOT=/workspace/cnla/skip-lens
R=/workspace/cnla/results/meansub_abl
GPU=${1:-2}

echo "[scale] prep 5k data (CPU)"
python cnla/prep_scale.py

for arm in raw meansub_norm; do
  echo "[scale] train ${arm}_5k (GPU $GPU)"
  CUDA_VISIBLE_DEVICES=$GPU python -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
    --parquet data/meansub/av_${arm}_5k.parquet --sidecar data/meansub/av_${arm}_5k.parquet \
    --save-dir ckpts/skiplens_${arm}_5k --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
    --num-steps 500 --batch-size 16 --gradient-accumulation-steps 4 --save-every 250 \
    --wandb-project skiplens-layer-ablation --wandb-group scale --wandb-name ${arm}_5k \
    > logs/train_${arm}_5k.log 2>&1
  [ -f ckpts/skiplens_${arm}_5k/iter_0000500/adapter_config.json ] || { echo "TRAIN_FAILED ${arm}"; tail -5 logs/train_${arm}_5k.log; exit 1; }
done

cd "$ROOT/evals"
echo "[scale] eval raw_5k"
CUDA_VISIBLE_DEVICES=$GPU python fedlayer_meansub_eval.py --ao-ckpt "$ROOT/ckpts/skiplens_raw_5k/iter_0000500" \
  --jsnap-dir /workspace/cnla/results/jlens_snap --evals-dir "$ROOT/evals/datasets_fed" \
  --fed-layers 62,42 --rollout-len 16 --out "$R/fed_raw5k.json" > "$R/eval_raw5k.log" 2>&1
echo "[scale] eval meansub_norm_5k"
CUDA_VISIBLE_DEVICES=$GPU python fedlayer_meansub_eval.py --ao-ckpt "$ROOT/ckpts/skiplens_meansub_norm_5k/iter_0000500" \
  --jsnap-dir /workspace/cnla/results/jlens_snap --evals-dir "$ROOT/evals/datasets_fed" \
  --fed-layers 62,42 --rollout-len 16 --normalize-first \
  --sub-mean-npz "$ROOT/data/meansub/mean_test_dir_by_layer.npz" \
  --out "$R/fed_meansubnorm5k.json" > "$R/eval_meansubnorm5k.log" 2>&1

python truncate_readouts.py 8 "$R/fed_raw5k.json" "$R/fed_meansubnorm5k.json" > /dev/null 2>&1
K=${ANTHROPIC_API_KEY_FALLBACK:-$ANTHROPIC_API_KEY}
for T in raw5k meansubnorm5k; do
  ANTHROPIC_API_KEY=$K python judge_fedlayer.py --in "$R/fed_${T}_8tok.json" \
    --out "$R/judged_${T}_8tok.json" --model claude-sonnet-5 >> "$R/judge_scale.log" 2>&1
done
echo SCALE_DONE >> "$R/judge_scale.log"
