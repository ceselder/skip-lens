#!/bin/bash
#SBATCH --job-name=skiplens_true_opd8_post
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export PYTHONPATH=$SRC
cd "$SRC"

python scripts/merge_eval_json.py \
  --inputs "$ROOT"/results/true_opd8/*_eval.json \
  --out "$ROOT/results/true_opd8/quality_inputs.json"
python -m evals.judge_opd_quality \
  --input "$ROOT/results/true_opd8/quality_inputs.json" \
  --out "$ROOT/results/true_opd8/quality_judged.json"
python -m evals.analyze_true_opd8 \
  --results "$ROOT/results/true_opd8" \
  --checkpoints "$ROOT/checkpoints" \
  --quality "$ROOT/results/true_opd8/quality_judged.json" \
  --out "$ROOT/results/true_opd8/analysis.json"
echo TRUE_OPD8_POSTPROCESS_DONE
