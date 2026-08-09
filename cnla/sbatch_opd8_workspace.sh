#!/bin/bash
#SBATCH --job-name=skiplens_opd8_workspace
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=36:00:00
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
mkdir -p "$ROOT/results/opd8_workspace"

latest_checkpoint() {
  find "$1" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' | sort | tail -n 1
}

opd_ckpt=$(latest_checkpoint "$ROOT/checkpoints/normal_opd8")
sft_ckpt=$(latest_checkpoint "$ROOT/checkpoints/normal_sft8_tokens")
test -n "$opd_ckpt" && test -n "$sft_ckpt"

# The exact six released prompt sets, paper read positions, workspace layer
# sweep, raw/Jacobian vectors, and same-distribution shuffled controls.
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; \
    python -m evals.workspace_readout --base-ckpt '$BASE' --av-ckpt '$opd_ckpt' \
      --sidecar '$opd_ckpt' --jdir '$JDIR' --layers 20,32,42,54,62 \
      --modes raw,jac,shuffle --samples 2 --max-new-tokens 24 \
      --out '$ROOT/results/opd8_workspace/workspace_opd8.json'" &
opd_pid=$!
srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=120G \
  bash -lc "source /workspace-vast/celeste/.keys.env; source $VENV/bin/activate; cd $SRC; \
    python -m evals.workspace_readout --base-ckpt '$BASE' --av-ckpt '$sft_ckpt' \
      --sidecar '$sft_ckpt' --jdir '$JDIR' --layers 20,32,42,54,62 \
      --modes raw,jac,shuffle --samples 2 --max-new-tokens 24 \
      --out '$ROOT/results/opd8_workspace/workspace_sft8_tokens.json'" &
sft_pid=$!
wait "$opd_pid"
wait "$sft_pid"

python -m evals.judge_workspace_readouts \
  --input "$ROOT/results/opd8_workspace/workspace_opd8.json" \
  --out "$ROOT/results/opd8_workspace/workspace_opd8_judged.json"
python -m evals.judge_workspace_readouts \
  --input "$ROOT/results/opd8_workspace/workspace_sft8_tokens.json" \
  --out "$ROOT/results/opd8_workspace/workspace_sft8_tokens_judged.json"
echo OPD8_WORKSPACE_DONE
