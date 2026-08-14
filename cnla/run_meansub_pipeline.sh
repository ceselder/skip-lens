#!/bin/bash
# Mean-subtraction ablation downstream pipeline. Waits for both 500-pair lenses to finish, then:
#   raw control : feed RAW h_l                   (skiplens_raw_500)
#   mean-sub    : feed h_l - mean_l_test         (skiplens_meansub_500, per-layer test means)
# Each fed layer scored vs its OWN local J-lens workspace. 8-token readouts, Sonnet-5 judge.
set -uo pipefail
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
ROOT=/workspace/cnla/skip-lens
R=/workspace/cnla/results/meansub_abl; mkdir -p "$R"
CKR="$ROOT/ckpts/skiplens_raw_500/iter_0000300"
CKM="$ROOT/ckpts/skiplens_meansub_500/iter_0000300"
JSNAP=/workspace/cnla/results/jlens_snap
EV="$ROOT/evals/datasets_fed"
MEAN="$ROOT/data/meansub/mean_test_by_layer.npz"
FED=62,55,48,42,34,26
K=${ANTHROPIC_API_KEY_FALLBACK:-$ANTHROPIC_API_KEY}

echo "[pipe] waiting for both final checkpoints..."
while [ ! -f "$CKR/adapter_config.json" ] || [ ! -f "$CKM/adapter_config.json" ]; do sleep 20; done
sleep 10
echo "[pipe] checkpoints ready -> evals"

cd "$ROOT/evals"
CUDA_VISIBLE_DEVICES=1 python fedlayer_meansub_eval.py --ao-ckpt "$CKR" \
  --jsnap-dir "$JSNAP" --evals-dir "$EV" --fed-layers "$FED" --rollout-len 16 \
  --out "$R/fed_raw500.json" > "$R/eval_raw.log" 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=2 python fedlayer_meansub_eval.py --ao-ckpt "$CKM" \
  --jsnap-dir "$JSNAP" --evals-dir "$EV" --fed-layers "$FED" --rollout-len 16 \
  --sub-mean-npz "$MEAN" \
  --out "$R/fed_meansub500.json" > "$R/eval_meansub.log" 2>&1 &
P2=$!
wait $P1; wait $P2
echo "[pipe] evals done -> truncate to 8 tok"

python truncate_readouts.py 8 "$R/fed_raw500.json" "$R/fed_meansub500.json"

for T in raw500 meansub500; do
  echo "[pipe] judging $T"
  ANTHROPIC_API_KEY=$K python judge_fedlayer.py --in "$R/fed_${T}_8tok.json" \
    --out "$R/judged_${T}_8tok.json" --model claude-sonnet-5
done
echo MEANSUB_PIPELINE_DONE
