#!/bin/bash
# 8-token re-analysis: truncate the (local-target) readouts to 8 tokens, then re-judge
# workspace/answer + coherence on the shorter readouts. API-only, fallback key.
set -uo pipefail
cd /workspace/cnla/skip-lens/evals
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/cnla/skip-lens
PY=/workspace/cnla_venv/bin/python
K=${ANTHROPIC_API_KEY_FALLBACK:-$ANTHROPIC_API_KEY}
R=/workspace/cnla/results/layerabl
$PY truncate_readouts.py 8 "$R/fed_L62train_local.json" "$R/fed_L42train_local.json" "$R/fed_L62curtok_local.json"
for T in L62train L42train L62curtok; do
  echo "=== 8tok workspace/answer: $T ==="
  ANTHROPIC_API_KEY=$K $PY judge_fedlayer.py --in "$R/fed_${T}_local_8tok.json" \
    --out "$R/judged_${T}_local_8tok.json" --model claude-sonnet-5
  echo "=== 8tok coherence: $T ==="
  ANTHROPIC_API_KEY=$K $PY judge_coherence.py --in "$R/fed_${T}_local_8tok.json" \
    --out "$R/coherence_${T}_8tok.json" --model claude-sonnet-5
done
echo ALL_8TOK_DONE
