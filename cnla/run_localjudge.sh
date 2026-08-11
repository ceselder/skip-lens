#!/bin/bash
# Judge the LOCAL-J-retargeted fed-layer readouts (each layer vs its own workspace).
# Uses the fallback Anthropic key if present (the low-prio key was rate-limiting).
set -uo pipefail
cd /workspace/cnla/skip-lens/evals
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/cnla/skip-lens
PY=/workspace/cnla_venv/bin/python
K=${ANTHROPIC_API_KEY_FALLBACK:-$ANTHROPIC_API_KEY}
R=/workspace/cnla/results/layerabl
for T in L62train L42train L62curtok; do
  echo "=== judging LOCAL: $T ==="
  ANTHROPIC_API_KEY=$K $PY judge_fedlayer.py --in "$R/fed_${T}_local.json" \
    --out "$R/judged_${T}_local.json" --model claude-sonnet-5
done
echo ALL_LOCALJUDGE_DONE
