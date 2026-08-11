#!/bin/bash
# L42-matched vs L62-mismatch, fed at L42, WITH vs WITHOUT Jacobian.
# Hypothesis: L62-trained-fed-L42 (esp. +Jacobian) -> high agree_JLENS (workspace), low agree_ANSWER;
#             L42-matched -> low agree_JLENS, high agree_ANSWER.
set -uo pipefail
cd /workspace/cnla/skip-lens/evals
source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
export PYTHONPATH=/workspace/cnla/skip-lens:/workspace/cnla/skip-lens/evals
export CUDA_VISIBLE_DEVICES=5
PY=/workspace/cnla_venv/bin/python
JDIR=/workspace/cnla/results/jlens_snap
EVD=/workspace/cnla/skip-lens/evals/datasets_fed
OUT=/workspace/cnla/results/l42_mismatch_eval; mkdir -p "$OUT"

for pair in "l42matched:/workspace/cnla/adapters/l42_matched" "l62mismatch:/workspace/cnla/adapters/l62_mismatch"; do
  name=${pair%%:*}; ckpt=${pair#*:}
  echo "=== FEDLAYER $name ($ckpt) fed@L42 raw+jac ==="
  $PY fedlayer_sweep_eval.py --ao-ckpt "$ckpt" --jdir "$JDIR" --jorigin 42 --jtarget 62 \
    --fed-layers 42 --evals-dir "$EVD" \
    --out "$OUT/${name}_raw.json" --out-jac "$OUT/${name}_jac.json" \
    --n-ao 4 --rollout-len 24 2>&1 | tail -4
done

echo "=== JUDGING (Sonnet-5): agree_JLENS [workspace] vs agree_ANSWER [surface] ==="
for f in l42matched_raw l42matched_jac l62mismatch_raw l62mismatch_jac; do
  echo "-- judge $f --"
  $PY judge_fedlayer.py --in "$OUT/$f.json" --out "$OUT/${f}_judged.json" --model claude-sonnet-5 2>&1 | tail -3
done
echo "ALL_DONE_L42_MISMATCH_EVAL"
