#!/bin/bash
# Test the regression lens on the already-trained single-vector arms, with no
# retraining. Both arms train on a token-pooled LOCAL transport; the only question
# is which fixed corpus-level operator best stands in for it at test time.
#
#   pooled_avg      sum_d w_d Jbar^(d) h42      the plain averaged Jacobian
#   regression_avg  sum_d w_d W*_d    h42      the ridge-fit maps, SAME weights
#   raw_h42         h42                         does the operator earn its keep
#   mean_only       a position-independent vector, the floor
#
# W* predicts the local transport the decoder actually trained on about twice as
# well as E[J] (centered cos 0.336 vs 0.165 on held-out rows, centered effective
# rank 29.7 vs 6.4), so for a TRAINED-DECODER readout it should be the better
# test-time vector. Note this is the opposite of what a logit-lens readout wants:
# there E[J] wins (+0.056 vs +0.038 span recall), because ridge spends its capacity
# on the large-norm directions of the transport rather than the small
# unembedding-readable component. Same operator, two readouts, opposite verdicts —
# which is the point of running both conditions through the same decoder.
#
# Weights are taken from the Jbar norms in both conditions, matching each arm's
# training target exactly.
set -u
cd /workspace/skip-lens/evals
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens:/workspace/skip-lens/evals
R=/workspace/results/multislot_eval
CONDS=pooled_avg,regression_avg,raw_h42,mean_only

run() {                     # run <arm> <ckpt-dir> <deep-from> <weight-mode> <sidecar>
  local arm=$1 ck=$2 df=$3 wm=$4 side=$5
  echo "[$arm] waiting for training to finish"
  while pgrep -f "save-dir ckpts/${ck}" >/dev/null; do sleep 120; done
  local CK
  CK=$(ls -d /workspace/skip-lens/ckpts/${ck}/iter_* 2>/dev/null | sort | tail -1)
  [ -z "$CK" ] && { echo "[$arm] no checkpoint, skipping"; return; }
  local G
  G=$(/workspace/gpu_wait.sh 70000)
  echo "[$arm] evaluating $CK on gpu $G (deep-from=$df, weights=$wm)"
  CUDA_VISIBLE_DEVICES=$G python pooled_single_eval.py \
    --av-ckpt "$CK" \
    --jbar-dir /workspace/results/offset_jlens \
    --reg-dir /workspace/results/regression_lens \
    --deep-from "$df" --weight-mode "$wm" \
    --template-yaml "$side" \
    --evals-dir /workspace/skip-lens/evals/datasets/official/evaluations \
    --conditions "$CONDS" --n-ao 4 --rollout-len 24 \
    --out "$R/${arm}_reg.json" || { echo "[$arm] EVAL FAILED"; return; }
  python judge_fedlayer.py --in "$R/${arm}_reg.json" \
    --out "$R/${arm}_reg_judged.json" --model claude-sonnet-5 || return
  python rescore_all.py --glob "$R/${arm}_reg_judged.json" \
    --out "$R/${arm}_reg_rescored.json"
}

run armI armI_pooled_single 0 uniform \
    /workspace/data/final/pooled1_train.parquet.nla_meta.yaml
run armK armK_deep_single 1 norm_eq \
    /workspace/data/final/deep1_train.parquet.nla_meta.yaml
echo "all regression-arm evals done."
