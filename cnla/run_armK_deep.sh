#!/bin/bash
# ARM K — arm I, minus the offset that was doing all the damage.
#
#   TRAIN  v = [ sum_{d>=1} w_d J_local^(d)(this context) ] @ h42[p]
#   TEST   v = [ sum_{d>=1} w_d Jbar^(d) ]                 @ h42[p]
#   w_d = 1/||Jbar^(d)||   (same weights both sides — a design choice, not a
#                           train/test mismatch)
#
# Why exclude offset 0. It is the ONLY offset where raw h42 already beats the
# Jacobian at predicting the training target (0.437 vs 0.395), and with
# ||Jbar^(0)||=30.8 against 6.6, 3.4, ... it dominates any uniform sum. That is
# exactly why arm I's uniform pooling LOSES to a raw-h42 control (0.366 vs
# 0.381) despite pooling doubling the train/test alignment over per-offset slots
# (0.145 -> 0.366). Drop offset 0 and norm-equalise, and the averaged Jacobian
# becomes additive over the activation: +0.074 alignment margin.
#
# And the readout-space evidence, which is what actually matters here: scoring
# each candidate vector through the unembedding against the tokens that really
# follow, with a shuffled-context control to subtract generic token frequency,
#
#   raw h42                    +0.019      (its raw 0.080 recall was 76% frequency)
#   Jbar^(0) h42               +0.037      (the paper's J-lens vector)
#   deep pooled, d>=1          +0.044      <- the most future content of any vector
#
# so the deep operator carries MORE context-specific future content than either
# the activation or the pooled J-lens vector. Its low cosine to the local
# transport (0.149) was cosine in 5120 dimensions being the wrong metric, not an
# absence of information. Centering made it slightly worse (+0.034), so the
# recurring British-English/EU basin is not a pure mean offset.
#
# K=1 for the reasons that killed the eight-slot family: averaging collapses
# slots toward each other (0.458 test collinearity where training saw 0.096;
# 0.86 vs 0.29 for the eight per-offset slots), and no scaling fix reaches it —
# preserving faint slots' magnitudes is WORSE than amplifying them (+8.65 vs
# +5.26 nats).
set -u
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens

T=/workspace/data/final/deep1_train.parquet
V=/workspace/data/final/deep1_val.parquet

echo "[armK] waiting for the deep-pooled dataset"
until [ -f "$T" ]; do sleep 60; done

N=$(python -c "import pyarrow.parquet as pq; print(max(200, pq.ParquetFile('$T').metadata.num_rows // 64))")
G=${FORCE_GPU:-$(/workspace/gpu_wait.sh 70000)}
echo "[armK] training on gpu $G for $N steps (n_slots=1, deep-only norm-equalised)"
CUDA_VISIBLE_DEVICES=$G python -m nla.train_sft --mode av \
  --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$T" --sidecar "$T" --n-slots 1 \
  --save-dir ckpts/armK_deep_single \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$N" --batch-size 16 --gradient-accumulation-steps 4 \
  --save-every 500 --heldout-parquet "$V" --heldout-rows 800 --heldout-every 250 \
  --sample-every 500 --n-samples 4 --sample-max-new-tokens 48 \
  --wandb-project skiplens-multislot --wandb-name armK_deep_single
echo "done training."

CK=$(ls -d ckpts/armK_deep_single/iter_* | sort | tail -1)
G=$(/workspace/gpu_wait.sh 70000)
echo "[armK] evaluating $CK on gpu $G"
cd /workspace/skip-lens/evals
export PYTHONPATH=/workspace/skip-lens:/workspace/skip-lens/evals
CUDA_VISIBLE_DEVICES=$G python pooled_single_eval.py \
  --av-ckpt "/workspace/skip-lens/$CK" \
  --jbar-dir /workspace/results/offset_jlens \
  --deep-from 1 --weight-mode norm_eq \
  --template-yaml "$T.nla_meta.yaml" \
  --evals-dir /workspace/skip-lens/evals/datasets/official/evaluations \
  --conditions pooled_avg,raw_h42,mean_only \
  --n-ao 4 --rollout-len 24 \
  --out /workspace/results/multislot_eval/armK.json
python judge_fedlayer.py --in /workspace/results/multislot_eval/armK.json \
  --out /workspace/results/multislot_eval/armK_judged.json --model claude-sonnet-5
python rescore_all.py --glob "/workspace/results/multislot_eval/armK_judged.json" \
  --out /workspace/results/multislot_eval/armK_rescored.json
