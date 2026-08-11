#!/bin/bash
# R-arm of the L42 workspace eval: feed R_42->62 . h_42 to each lens, keeping the FIXED
# J-lens agree_jlens reference (jl_top from real J in jlens_snap). Adds 2 conditions:
# {L42-matched, L62-mismatch} x {feed=R}.  (Spike R: 24 prompts, L42 only.)
set -uo pipefail
cd /workspace/cnla/skip-lens/evals
source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
export PYTHONPATH=/workspace/cnla/skip-lens:/workspace/cnla/skip-lens/evals
export CUDA_VISIBLE_DEVICES=6
PY=/workspace/cnla_venv/bin/python
JDIR=/workspace/cnla/results/jlens_snap        # FIXED jl_top reference (real J)
RDIR=/workspace/cnla/results/rlens_jdir        # R feed matrices (R named J_L42_to_L62.npy)
EVD=/workspace/cnla/skip-lens/evals/datasets_fed
OUT=/workspace/cnla/results/l42_mismatch_eval; mkdir -p "$OUT"

for pair in "l42matched:/workspace/cnla/adapters/l42_matched" "l62mismatch:/workspace/cnla/adapters/l62_mismatch"; do
  name=${pair%%:*}; ckpt=${pair#*:}
  echo "=== R-ARM $name fed@L42 (feed=R, reference=J) ==="
  $PY fedlayer_sweep_eval.py --ao-ckpt "$ckpt" --jdir "$JDIR" --feed-jdir "$RDIR" --jorigin 42 --jtarget 62 \
    --fed-layers 42 --evals-dir "$EVD" \
    --out "$OUT/${name}_Rraw_discard.json" --out-jac "$OUT/${name}_R.json" \
    --n-ao 4 --rollout-len 24 2>&1 | tail -4
done

echo "=== JUDGING R arm (Sonnet-5) ==="
for f in l42matched_R l62mismatch_R; do
  echo "-- judge $f --"
  $PY judge_fedlayer.py --in "$OUT/$f.json" --out "$OUT/${f}_judged.json" --model claude-sonnet-5 2>&1 | tail -3
done
echo "ALL_DONE_R_ARM"
