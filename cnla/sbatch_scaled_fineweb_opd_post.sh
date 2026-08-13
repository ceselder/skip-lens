#!/bin/bash
#SBATCH --job-name=skiplens_opd_scale_post
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/workspace-vast/celeste/skip-lens-opd/logs/%x_%j.out
set -euo pipefail

ROOT=${ROOT:-/workspace-vast/celeste/skip-lens-opd}
SRC=$ROOT/src
VENV=$ROOT/venv
CKPTS=$ROOT/checkpoints/fineweb_t1_scaled_100k
RESULTS=$ROOT/results/fineweb_t1_scaled_100k

source /workspace-vast/celeste/.keys.env
source "$VENV/bin/activate"
export PYTHONPATH=$SRC
cd "$SRC"

python -m evals.analyze_scaled_opd \
  --results "$RESULTS/evals" --checkpoints "$CKPTS" \
  --out "$RESULTS/scaled_opd_analysis.json"
python -m evals.prepare_scaled_opd_quality \
  --analysis "$RESULTS/scaled_opd_analysis.json" --results "$RESULTS/evals" \
  --out "$RESULTS/quality_inputs.json"
python -m evals.judge_opd_quality \
  --input "$RESULTS/quality_inputs.json" \
  --out "$RESULTS/quality_judged.json"
python -m evals.analyze_scaled_opd \
  --results "$RESULTS/evals" --checkpoints "$CKPTS" \
  --quality "$RESULTS/quality_judged.json" \
  --out "$RESULTS/scaled_opd_analysis.json"
python scripts/plot_scaled_opd.py \
  --analysis "$RESULTS/scaled_opd_analysis.json" \
  --out-dir "$RESULTS/report"

echo SCALED_FINEWEB_OPD_POST_DONE
