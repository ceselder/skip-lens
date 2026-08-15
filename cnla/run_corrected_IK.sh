#!/bin/bash
# Corrected arm I and arm K. Same design as before; three bugs fixed and the idle
# data brought in. One epoch, as before.
#
# 1. UNIFORM weights, not 1/||Jbar^(d)||. The old weights came from the AVERAGED
#    matrix norms but multiplied the LOCAL transports, which decay ~4x more slowly,
#    so they over-corrected and INVERTED the ordering: offset 14 became the largest
#    contributor (15.5) and offset 1 the smallest (3.98), when naturally offset 1 is
#    largest (26.4). The local transports span only 4.29x on their own, so no
#    weighting is needed.
#
# 2. RESPONSE ALIGNED TO THE POOLED OFFSETS. Offset d is h62[p+d], which predicts
#    roll[d]. Old arm K pooled offsets 1..15 against roll[0:8], so its vector said
#    nothing about the first token it had to emit while carrying content about eight
#    tokens it was never asked for. Now:
#       arm I  pools 0..7  -> targets roll[0:8]
#       arm K  pools 1..8  -> targets roll[1:9]   (offset 0 excluded as intended)
#    Both produce 8-token responses, so the two arms stay comparable.
#
# 3. ALL 7 SHARDS (~395k rows) instead of shards 0-3 (248k) - the +59% that was
#    collected and never used. Still one epoch.
#
# Eval passes --n-offsets equal to each arm's pool_upto so the test-time operator is
# summed over exactly the offsets the arm trained on, and runs pooled_avg against
# regression_avg (W*), raw_h42 and the position-independent mean_only floor.
set -u
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens
SHARDS="/workspace/data/spans_jvp/shard_[0-9]_jvp.parquet"
R=/workspace/results/multislot_eval

build() {                      # build <tag> <deep-from> <pool-upto>
  local tag=$1 lo=$2 hi=$3
  local T=/workspace/data/final/${tag}_train.parquet
  if [ ! -f "$T" ]; then
    echo "=== building $tag (pool offsets $lo..$((hi-1)), target roll[$lo:$hi]) ==="
    python -m pretrain.build_pooled_single \
      --shards-glob "$SHARDS" \
      --out-train "$T" --out-val "/workspace/data/final/${tag}_val.parquet" \
      --deep-from "$lo" --pool-upto "$hi" --weight-mode uniform || return 1
  fi
}

train_eval() {                 # train_eval <tag> <ckpt> <deep-from> <pool-upto>
  local tag=$1 ck=$2 lo=$3 hi=$4
  local T=/workspace/data/final/${tag}_train.parquet
  local V=/workspace/data/final/${tag}_val.parquet
  local N G CK
  N=$(python -c "import pyarrow.parquet as pq; print(max(200, pq.ParquetFile('$T').metadata.num_rows // 64))")
  G=$(/workspace/gpu_wait.sh 70000)
  echo "=== training $tag on gpu $G for $N steps (one epoch, n_slots=1) ==="
  CUDA_VISIBLE_DEVICES=$G python -m nla.train_sft --mode av \
    --base-ckpt Qwen/Qwen3.6-27B \
    --parquet "$T" --sidecar "$T" --n-slots 1 \
    --save-dir "ckpts/$ck" \
    --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
    --num-steps "$N" --batch-size 16 --gradient-accumulation-steps 4 \
    --save-every 1000 --heldout-parquet "$V" --heldout-rows 800 --heldout-every 250 \
    --sample-every 1000 --n-samples 4 --sample-max-new-tokens 32 \
    --wandb-project skiplens-multislot --wandb-name "$tag" || return 1
  CK=$(ls -d /workspace/skip-lens/ckpts/$ck/iter_* | sort | tail -1)
  G=$(/workspace/gpu_wait.sh 70000)
  echo "=== evaluating $tag ($CK) on gpu $G ==="
  cd /workspace/skip-lens/evals
  export PYTHONPATH=/workspace/skip-lens:/workspace/skip-lens/evals
  CUDA_VISIBLE_DEVICES=$G python pooled_single_eval.py \
    --av-ckpt "$CK" \
    --jbar-dir /workspace/results/offset_jlens \
    --reg-dir /workspace/results/regression_lens \
    --deep-from "$lo" --n-offsets "$hi" --weight-mode uniform \
    --template-yaml "$T.nla_meta.yaml" \
    --evals-dir /workspace/skip-lens/evals/datasets/official/evaluations \
    --conditions pooled_avg,regression_avg,raw_h42,mean_only \
    --n-ao 4 --rollout-len 24 --out "$R/${tag}.json" || { cd /workspace/skip-lens; return 1; }
  python judge_fedlayer.py --in "$R/${tag}.json" \
    --out "$R/${tag}_judged.json" --model claude-sonnet-5
  python rescore_all.py --glob "$R/${tag}_judged.json" --out "$R/${tag}_rescored.json"
  cd /workspace/skip-lens
}

build armI2 0 8 && build armK2 1 9 || { echo "BUILD FAILED"; exit 1; }
train_eval armI2 armI2_pool0to8 0 8
train_eval armK2 armK2_pool1to9 1 9
echo "corrected I/K done."
