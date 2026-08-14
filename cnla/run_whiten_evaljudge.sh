#!/bin/bash
# Resume whitening after the eval got SIGKILL'd (node contention): re-run eval + truncate + judge only
# (checkpoint skiplens_whiten_500 already trained, whiten_test_stats.npz already computed).
set -uo pipefail
cd /workspace/cnla/skip-lens
source /workspace/cnla_venv/bin/activate
source /workspace/.keys.env
export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/cnla/skip-lens
ROOT=/workspace/cnla/skip-lens
R=/workspace/cnla/results/meansub_abl
GPU=${1:-6}
CKW="$ROOT/ckpts/skiplens_whiten_500/iter_0000300"

cd "$ROOT/evals"
CUDA_VISIBLE_DEVICES=$GPU python fedlayer_meansub_eval.py --ao-ckpt "$CKW" \
  --jsnap-dir /workspace/cnla/results/jlens_snap --evals-dir "$ROOT/evals/datasets_fed" \
  --fed-layers 62,55,48,42,34,26 --rollout-len 16 \
  --whiten-npz "$ROOT/data/meansub/whiten_test_stats.npz" \
  --out "$R/fed_whiten500.json" > "$R/eval_whiten.log" 2>&1
[ -f "$R/fed_whiten500.json" ] || { echo EVAL_FAILED; tail -6 "$R/eval_whiten.log"; exit 1; }
python truncate_readouts.py 8 "$R/fed_whiten500.json"
K=${ANTHROPIC_API_KEY_FALLBACK:-$ANTHROPIC_API_KEY}
ANTHROPIC_API_KEY=$K python judge_fedlayer.py --in "$R/fed_whiten500_8tok.json" \
  --out "$R/judged_whiten500_8tok.json" --model claude-sonnet-5
echo WHITEN_EVALJUDGE_DONE
