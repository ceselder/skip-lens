#!/bin/bash
# ARM G — the two-J design with SYMMETRIC source-space centering.
#
#   TRAIN slots[d] = Jbar^(d)_{62->63} @ (h62[p] - mu_62)
#   TEST  slots[d] = Jbar^(d)_{42->63} @ (h42[p] - mu_42)
#
# Each side is centered in ITS OWN source space. This is what the earlier
# "centered" condition should have been: that one subtracted mu_42 at TEST time
# only, against a decoder trained on uncentered vectors, so its poor score
# (0.218) measured an unseen transformation rather than the hypothesis.
#
# Centering is not a small correction here: ||mu_42||=65.7 against h42 norms ~90,
# and ||mu_62||=188 against h62 norms ~355 — over half of a typical activation.
# The two spaces also differ ~28x in eigenvalue scale, which is why the
# standardization has to be per-space rather than shared.
#
# Whitening is deliberately NOT enabled here: Sigma^{-1/2} has condition ~1e4-2e4
# after ridge, so it amplifies the lowest-variance (plausibly noise) directions
# ~100x relative to the top ones. Run that as a separate variant so we can tell
# which change did what.
set -u
cd /workspace/skip-lens
source /workspace/venv/bin/activate
source /workspace/.keys.env
export PYTHONPATH=/workspace/skip-lens

JD=/workspace/results/offset_jlens_last
T=/workspace/data/final/armG_centered_train.parquet
V=/workspace/data/final/armG_centered_val.parquet

# the eval reads its centering assets from --jbar-dir, so make them visible there
for f in hbar_L42.npy hbar_L62.npy whiten_L42.npy whiten_L62.npy; do
  [ -f "$JD/$f" ] || cp "/workspace/results/offset_jlens/$f" "$JD/$f"
done

if [ ! -f "$T" ]; then
  G=${FORCE_GPU:-$(/workspace/gpu_wait.sh 20000)}
  echo "[armG] building centered train data on gpu $G"
  CUDA_VISIBLE_DEVICES=$G python pretrain/build_twoJ_train.py \
    --raw-glob "/workspace/data/spans_raw/shard_[0-3].parquet" \
    --jbar-dir "$JD" --src-layer 62 --src-col act_L62 --tgt-layer 63 \
    --k-slots 8 --center --stdz-dir /workspace/results/offset_jlens \
    --out-train "$T" --out-val "$V" || exit 1
fi

N=$(python -c "import pyarrow.parquet as pq; print(max(200, pq.ParquetFile('$T').metadata.num_rows // 64))")
G=${FORCE_GPU:-$(/workspace/gpu_wait.sh 70000)}
echo "[armG] training on gpu $G for $N steps"
CUDA_VISIBLE_DEVICES=$G python -m nla.train_sft --mode av \
  --base-ckpt Qwen/Qwen3.6-27B \
  --parquet "$T" --sidecar "$T" --n-slots 8 \
  --save-dir ckpts/multislot_armG_centered \
  --use-lora --lora-r 64 --lora-alpha 16 --lora-scope all \
  --num-steps "$N" --batch-size 16 --gradient-accumulation-steps 4 \
  --save-every 500 --heldout-parquet "$V" --heldout-rows 800 --heldout-every 250 \
  --sample-every 500 --n-samples 4 --sample-max-new-tokens 48 \
  --wandb-project skiplens-multislot --wandb-name armG_centered_62to63
echo "[armG] done — evaluate with: --jbar-dir $JD --conditions centered,per_offset,diff"
