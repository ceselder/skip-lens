#!/bin/bash
# Redo the mean-subtraction ablation the RIGHT way: normalize each activation, take the mean in unit
# space, subtract that. Train skiplens_meansub_norm_500 (h/||h|| - mean_dir_L62), eval feeding
# h_l/||h_l|| - mean_dir_l_test, 8-tok, judge. Compare to the existing raw_500 control.
set -uo pipefail
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
ROOT=/workspace/cnla/skip-lens
R=/workspace/cnla/results/meansub_abl
GPU=${1:-1}
CKM="$ROOT/ckpts/skiplens_meansub_norm_500/iter_0000300"

echo "[pipe] training normed-meansub lens on gpu $GPU"
CUDA_VISIBLE_DEVICES=$GPU python -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
  --parquet data/meansub/av_meansub_norm_500.parquet --sidecar data/meansub/av_meansub_norm_500.parquet \
  --save-dir ckpts/skiplens_meansub_norm_500 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps 300 --batch-size 16 --gradient-accumulation-steps 4 --save-every 100 \
  --wandb-project skiplens-layer-ablation --wandb-group meansub --wandb-name meansub_norm_500 \
  > logs/train_meansub_norm_500.log 2>&1
[ -f "$CKM/adapter_config.json" ] || { echo TRAIN_FAILED; tail -5 logs/train_meansub_norm_500.log; exit 1; }
echo "[pipe] trained -> eval (normalize-first)"

cd "$ROOT/evals"
CUDA_VISIBLE_DEVICES=$GPU python fedlayer_meansub_eval.py --ao-ckpt "$CKM" \
  --jsnap-dir /workspace/cnla/results/jlens_snap --evals-dir "$ROOT/evals/datasets_fed" \
  --fed-layers 62,55,48,42,34,26 --rollout-len 16 --normalize-first \
  --sub-mean-npz "$ROOT/data/meansub/mean_test_dir_by_layer.npz" \
  --out "$R/fed_meansubnorm500.json" > "$R/eval_meansubnorm.log" 2>&1
echo "[pipe] eval done -> truncate + judge"
python truncate_readouts.py 8 "$R/fed_meansubnorm500.json"
K=${ANTHROPIC_API_KEY_FALLBACK:-$ANTHROPIC_API_KEY}
ANTHROPIC_API_KEY=$K python judge_fedlayer.py --in "$R/fed_meansubnorm500_8tok.json" \
  --out "$R/judged_meansubnorm500_8tok.json" --model claude-sonnet-5
echo MEANSUB_NORM_DONE
