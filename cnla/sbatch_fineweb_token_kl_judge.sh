#!/bin/bash
#SBATCH --job-name=skiplens_token_kl_judge
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
RESULTS=$ROOT/results/fineweb_token_kl

source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export PYTHONPATH=$SRC
cd "$SRC"

python scripts/merge_eval_json.py \
  --inputs "$RESULTS"/fineweb_token_kl_*_eval.json \
  --out "$RESULTS/quality_inputs.json"
python -m evals.judge_opd_quality \
  --input "$RESULTS/quality_inputs.json" \
  --out "$RESULTS/quality_judged.json"
echo FINEWEB_TOKEN_KL_JUDGE_DONE
