#!/bin/bash
# Evaluate each arm the moment its training finishes — so no checkpoint sits
# trained-but-unmeasured again (the frozen arm did for ~16h).
#
# Per arm: the TRAINED construction is evaluated first, then `diff` (the only
# construction that has reached baseline parity anywhere) as a cross-check.
#   armE  trained on Jbar_{62->63} slots   -> test family 42->63, per_offset
#   armF  trained on Jbar_{42->62} slots   -> SAME family at test (matched),
#                                             so per_offset here IS the ceiling
#   armG  trained on CENTERED 62->63 slots -> test family 42->63, `centered`
#
# breakdown_multislot.py now reports the degenerate-output fraction and the
# 0/1/2 score histogram alongside every mean, because a judged mean without
# those is uninterpretable (frozen per_offset: 0.439 with 47% degenerate).
set -u
cd /workspace/skip-lens/evals
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens:/workspace/skip-lens/evals
R=/workspace/results/multislot_eval
LOG=/workspace/logs/auto_eval.log
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

run_arm () {           # $1=arm  $2=ckpt-dir  $3=jbar-dir  $4=conditions
  local arm=$1 ckroot=$2 jd=$3 conds=$4
  local out="$R/${arm}.json"
  [ -f "$R/${arm}_breakdown.json" ] && { log "$arm already evaluated"; return; }
  log "$arm: waiting for training to finish"
  until grep -q "^done\.$" "/workspace/skip-lens/logs/train_${arm}.log" 2>/dev/null; do
    sleep 180
  done
  local ck
  ck=$(ls -d ${ckroot}/iter_* 2>/dev/null | sort | tail -1)
  [ -z "$ck" ] && { log "$arm: NO CHECKPOINT under $ckroot"; return; }
  local G
  G=$(/workspace/gpu_wait.sh 70000)
  log "$arm: evaluating $ck on gpu $G (conditions: $conds)"
  CUDA_VISIBLE_DEVICES=$G python multislot_fed_eval.py \
    --av-ckpt "$ck" --jbar-dir "$jd" --k-slots 8 \
    --src-layer 42 --tgt-layer $( [ "$jd" = "/workspace/results/offset_jlens" ] && echo 62 || echo 63 ) \
    --evals-dir /workspace/skip-lens/evals/datasets/official/evaluations \
    --conditions "$conds" --n-ao 4 --rollout-len 24 --out "$out" \
    >> "$LOG" 2>&1 || { log "$arm: EVAL FAILED"; return; }
  python judge_fedlayer.py --in "$out" --out "$R/${arm}_judged.json" \
    --model claude-sonnet-5 >> "$LOG" 2>&1
  python breakdown_multislot.py --judged "$R/${arm}_judged.json" \
    --out "$R/${arm}_breakdown.json" --examples 4 2>&1 | tee -a "$LOG"
  log "$arm: DONE"
}

CK=/workspace/skip-lens/ckpts
JD_PEN=/workspace/results/offset_jlens          # 42->62 family
JD_LAST=/workspace/results/offset_jlens_last    # 42->63 and 62->63 families

run_arm armF "$CK/multislot_armF_matched"  "$JD_PEN"  "per_offset,diff"
run_arm armG "$CK/multislot_armG_centered" "$JD_LAST" "centered,per_offset,diff"
run_arm armE "$CK/multislot_armE_twoJ"     "$JD_LAST" "per_offset,diff"
log "all arms evaluated"
