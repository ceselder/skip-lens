#!/bin/bash
# B300 per-checkpoint skip-lens eval: full harness (FVE sweep + coherence + greedy-logprob +
# A.6 readouts) then A.6 judging on 3 layers. Usage: run_skiplens_one.sh <av_dir> <gpu> <tag>
set -uo pipefail
AVDIR=$1; GPU=$2; TAG=$3
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=$PWD HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1
echo "=== skiplens eval $TAG | av=$AVDIR | gpu=$GPU ==="
CUDA_VISIBLE_DEVICES=$GPU python -m evals.skiplens_scaling_eval \
  --av-ckpt "$AVDIR" --ar-ckpt /workspace/cnla/ckpts/ar_L62/iter_0001500 \
  --sidecar /workspace/cnla/data/sidecar_ar_L62_train.parquet \
  --ctx-parquet data/cnla/collect_L62.parquet --a6-dir evals/datasets \
  --fed-layers 62,48,34,26,18,10 --n-fve 64 --n-a6 10 --tag "$TAG" \
  --out results/skiplens_$TAG.json
# A.6 judging (CPU/Anthropic only) on 3 representative layers; run from evals/ so `_llm` resolves.
cd /workspace/cnla/skip-lens/evals
for L in 62 34 10; do
  F=/workspace/cnla/skip-lens/results/a6_readouts_$TAG/readouts_L$L.json
  [ -f "$F" ] && python judge_intermediates.py --readouts "$F" --ks 1,2,4 --n_decoy 2 --tag ${TAG}_L$L 2>&1 | grep -aE "mean discrimination|saved" | tail -2
done
echo "SKIPLENS_ONE_DONE $TAG"
