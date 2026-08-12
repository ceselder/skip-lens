#!/bin/bash
# Autonomous: wait for AV-12 SFT to finish, then run the full skip-lens eval (workspace FVE sweep +
# coherence + A.6 readouts/judging) on its final checkpoint, using the SAME refs as the scaled-
# futurelens baseline (results/skiplens_scaled_iter_0006813.json) so the comparison is apples-to-apples.
cd /workspace/cnla/skip-lens
LOG=logs/av12_eval.log
echo "AUTO_EVAL_START $(date)" >> "$LOG"
# 1) wait for training to finish (process gone)
while pgrep -f "[a]v12_L62" >/dev/null 2>&1; do sleep 120; done
CK=$(ls -d ckpts/av12_L62/iter_* 2>/dev/null | sort -V | tail -1)
echo "AV12_TRAIN_DONE, evaluating ckpt=$CK $(date)" >> "$LOG"
if [ -z "$CK" ]; then echo "AUTO_EVAL_ABORT: no av12 checkpoint $(date)" >> "$LOG"; exit 1; fi
# 2) run the skiplens eval on gpu2 (freed when AV-12 finished)
bash cnla/run_skiplens_one.sh "$CK" 2 av12 >> "$LOG" 2>&1
echo "AV12_EVAL_DONE $(date)" >> "$LOG"
