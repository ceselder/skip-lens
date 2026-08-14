#!/bin/bash
# Whitening ablation end-to-end: ZCA-whiten L62 train dist -> train futurelens -> feed ZCA-whitened
# per-layer test activations -> judge. Compare to raw / normed-mean-sub on the same 500-pair budget.
set -uo pipefail
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
ROOT=/workspace/cnla/skip-lens
R=/workspace/cnla/results/meansub_abl
GPU=${1:-1}
CKW="$ROOT/ckpts/skiplens_whiten_500/iter_0000300"

echo "[whiten] 1/5 train-side ZCA + whitened parquet (CPU)"
python cnla/prep_whiten_train.py

echo "[whiten] 2/5 test-side per-layer ZCA stats (GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU python evals/whiten_test_stats.py

echo "[whiten] 3/5 train skiplens_whiten_500 (GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU python -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
  --parquet data/meansub/av_whiten_500.parquet --sidecar data/meansub/av_whiten_500.parquet \
  --save-dir ckpts/skiplens_whiten_500 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps 300 --batch-size 16 --gradient-accumulation-steps 4 --save-every 100 \
  --wandb-project skiplens-layer-ablation --wandb-group meansub --wandb-name whiten_500 \
  > logs/train_whiten_500.log 2>&1
[ -f "$CKW/adapter_config.json" ] || { echo TRAIN_FAILED; tail -5 logs/train_whiten_500.log; exit 1; }

echo "[whiten] 4/5 eval (feed ZCA-whitened activation)"
cd "$ROOT/evals"
CUDA_VISIBLE_DEVICES=$GPU python fedlayer_meansub_eval.py --ao-ckpt "$CKW" \
  --jsnap-dir /workspace/cnla/results/jlens_snap --evals-dir "$ROOT/evals/datasets_fed" \
  --fed-layers 62,55,48,42,34,26 --rollout-len 16 \
  --whiten-npz "$ROOT/data/meansub/whiten_test_stats.npz" \
  --out "$R/fed_whiten500.json" > "$R/eval_whiten.log" 2>&1

echo "[whiten] 5/5 truncate + judge"
python truncate_readouts.py 8 "$R/fed_whiten500.json"
K=${ANTHROPIC_API_KEY_FALLBACK:-$ANTHROPIC_API_KEY}
ANTHROPIC_API_KEY=$K python judge_fedlayer.py --in "$R/fed_whiten500_8tok.json" \
  --out "$R/judged_whiten500_8tok.json" --model claude-sonnet-5
echo WHITEN_DONE
