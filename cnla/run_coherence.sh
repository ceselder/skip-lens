#!/bin/bash
# Coherence-judge the fed-layer readouts (L62-trained, L42-trained, current-token). API-only, fallback key.
set -uo pipefail
cd /workspace/cnla/skip-lens/evals
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/cnla/skip-lens
PY=/workspace/cnla_venv/bin/python
K=${ANTHROPIC_API_KEY_FALLBACK:-$ANTHROPIC_API_KEY}
R=/workspace/cnla/results/layerabl
for T in L62train L42train L62curtok; do
  echo "=== coherence: $T ==="
  ANTHROPIC_API_KEY=$K $PY judge_coherence.py --in "$R/fed_${T}.json" \
    --out "$R/coherence_${T}.json" --model claude-sonnet-5
done
echo ALL_COHERENCE_DONE
