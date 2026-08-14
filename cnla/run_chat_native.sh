#!/bin/bash
# Chat-native rebuild (empty-think anchor): collect chat-native training pairs -> train raw-injection
# futurelens -> eval chat-native (--chat-template) -> judge. Establishes the chat-native workspace-
# agreement baseline on the CORRECT substrate (train & eval both harvest at the assistant anchor).
set -uo pipefail
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
ROOT=/workspace/cnla/skip-lens
R=/workspace/cnla/results/meansub_abl
GPU=${1:-1}
CK="$ROOT/ckpts/skiplens_chat_500/iter_0000300"

echo "[chat] 1/4 collect chat-native pairs (GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU python cnla/collect_chat_native.py

echo "[chat] 2/4 train skiplens_chat_500 (GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU python -m nla.train_sft --mode av --base-ckpt Qwen/Qwen3.6-27B \
  --parquet data/meansub/av_chat_500.parquet --sidecar data/meansub/av_chat_500.parquet \
  --save-dir ckpts/skiplens_chat_500 --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps 300 --batch-size 16 --gradient-accumulation-steps 4 --save-every 100 \
  --wandb-project skiplens-layer-ablation --wandb-group chat-native --wandb-name chat_500 \
  > logs/train_chat_500.log 2>&1
[ -f "$CK/adapter_config.json" ] || { echo TRAIN_FAILED; tail -5 logs/train_chat_500.log; exit 1; }

echo "[chat] 3/4 eval chat-native (--chat-template)"
cd "$ROOT/evals"
CUDA_VISIBLE_DEVICES=$GPU python fedlayer_meansub_eval.py --ao-ckpt "$CK" \
  --jsnap-dir /workspace/cnla/results/jlens_snap --evals-dir "$ROOT/evals/datasets_fed" \
  --fed-layers 62,48,42,34 --rollout-len 16 --n-ao 4 --chat-template \
  --out "$R/fed_chat500.json" > "$R/eval_chatnative.log" 2>&1

echo "[chat] 4/4 truncate + judge"
python truncate_readouts.py 8 "$R/fed_chat500.json"
K=${ANTHROPIC_API_KEY_FALLBACK:-$ANTHROPIC_API_KEY}
ANTHROPIC_API_KEY=$K python judge_fedlayer.py --in "$R/fed_chat500_8tok.json" \
  --out "$R/judged_chat500_8tok.json" --model claude-sonnet-5
echo CHAT_NATIVE_DONE
