#!/bin/bash
# ARM I — the user's proposal. One vector, token-pooled both sides, and the ONLY
# train/test difference is the expectation over contexts:
#
#   TRAIN  v = [ sum_d J_local^(d)(this context) ] @ h42[p]   per-example gradient
#   TEST   v = [ sum_d Jbar^(d) ]              @ h42[p]   AVERAGED gradient
#                                                          ( = the J-lens vector )
#
# Why this is the best-motivated arm so far, from tonight's measurements:
#   * K=1 removes the per-slot norm-matching pathology that cost +5.80 nats
#     (test-time deep slots carry 0.2-3% of the real state's norm and were being
#     amplified ~500x back up to ||h_p||);
#   * both sides are the same operator shape, so the averaging is the only
#     variable — every earlier arm also changed the object type, which is the
#     confound that sent four hypotheses down blind alleys;
#   * the only configuration in this study that reads well is already K=1
#     (armC, 0.561), and it emits 8 tokens from one vector.
#
# Eval includes two controls a single-vector lens needs: raw_h42 (the transport
# only beats raw h42 by 0.585 vs 0.521 in cosine, so it must earn its keep) and
# mean_only (a position-independent vector, since cos(real state, mu_62) ~ 0.5
# means a mean-shaped readout can look deceptively plausible).
set -u
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens

T=/workspace/data/final/pooled1_train.parquet
V=/workspace/data/final/pooled1_val.parquet

echo "[armI] waiting for the pooled-single dataset"
until [ -f "$T" ]; do sleep 60; done

N=$(python -c "import pyarrow.parquet as pq; print(max(200, pq.ParquetFile('$T').metadata.num_rows // 64))")
G=${FORCE_GPU:-$(/workspace/gpu_wait.sh 70000)}
echo "[armI] training on gpu $G for $N steps (n_slots=1)"
CUDA_VISIBLE_DEVICES=$G python -m nla.train_sft --mode av \
  --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$T" --sidecar "$T" --n-slots 1 \
  --save-dir ckpts/armI_pooled_single \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$N" --batch-size 16 --gradient-accumulation-steps 4 \
  --save-every 500 --heldout-parquet "$V" --heldout-rows 800 --heldout-every 250 \
  --sample-every 500 --n-samples 4 --sample-max-new-tokens 48 \
  --wandb-project skiplens-multislot --wandb-name armI_pooled_single
echo "done."

CK=$(ls -d ckpts/armI_pooled_single/iter_* | sort | tail -1)
G=$(/workspace/gpu_wait.sh 70000)
echo "[armI] evaluating $CK on gpu $G"
cd /workspace/skip-lens/evals
export PYTHONPATH=/workspace/skip-lens:/workspace/skip-lens/evals
CUDA_VISIBLE_DEVICES=$G python pooled_single_eval.py \
  --av-ckpt "/workspace/skip-lens/$CK" \
  --jbar-dir /workspace/results/offset_jlens \
  --evals-dir /workspace/skip-lens/evals/datasets/official/evaluations \
  --conditions pooled_avg,raw_h42,mean_only \
  --n-ao 4 --rollout-len 24 \
  --out /workspace/results/multislot_eval/armI.json
python judge_fedlayer.py --in /workspace/results/multislot_eval/armI.json \
  --out /workspace/results/multislot_eval/armI_judged.json --model claude-sonnet-5
python breakdown_multislot.py --judged /workspace/results/multislot_eval/armI_judged.json \
  --out /workspace/results/multislot_eval/armI_breakdown.json --examples 5
