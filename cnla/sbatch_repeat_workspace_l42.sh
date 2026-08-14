#!/bin/bash
#SBATCH --job-name=skiplens_repeat_workspace
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
BASE=${BASE:-Qwen/Qwen3.6-27B}
JDIR=${JDIR:-/workspace-vast/celeste/multi-token-jlens-nla-lastlayer/results/jlens_official}
source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export HF_HOME=/workspace-vast/pretrained_ckpts
export HF_TOKEN_PATH=/workspace-vast/celeste/.cache/huggingface/token
export PYTHONPATH=$SRC

OUT=$ROOT/results/repeat_workspace_l42
CKPT=$ROOT/checkpoints/repeat_sft8_tokens/iter_0000917
mkdir -p "$OUT"
test -f "$CKPT/adapter_config.json"

# This checkpoint was trained only on CSPRNG word-sequence repetition data.
# decoder block 42's output is h_42 (= HF hidden_states[43]); h_62 is the
# in-distribution depth control for the same L62-trained lens.
python -m evals.workspace_readout \
  --base-ckpt "$BASE" --av-ckpt "$CKPT" --sidecar "$CKPT" --jdir "$JDIR" \
  --layers 42,62 --modes raw --samples 2 --max-new-tokens 24 \
  --temperature 0.8 --seed 0 \
  --out "$OUT/repeat_sft8_workspace.json"

python -m evals.judge_workspace_readouts \
  --input "$OUT/repeat_sft8_workspace.json" \
  --out "$OUT/repeat_sft8_workspace_judged.json"

echo REPEAT_WORKSPACE_L42_DONE
