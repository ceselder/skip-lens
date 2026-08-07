# skip-lens

Train, serve, and evaluate the **naive multi-token J-lens** (a.k.a. *future-lens*) on
`Qwen/Qwen3.6-27B` — plus optional **NLA reconstruction-RL** on top of it.

A future-lens reads a residual-stream activation `h_ℓ` and writes, in natural language, what the
model is about to produce from that point. The **naive** variant is trained purely to reproduce the
document's own continuation from a penultimate-layer (L62) activation; the **NLA** variant then
RL-autoencodes that reader so its readout reconstructs the activation. This repo bundles the whole
pipeline: pretraining, RL, evals, and an interactive lens playground.

Built on [easyNLA](https://github.com/asherps/EasyNLA) (Natural Language Autoencoders) — the `nla/`
package is the shared library and also carries the RL trainer.

## Layout

```
nla/         # shared library (from easyNLA): schema, injection, models, utils, datagen,
             #   train_sft.py (SFT), and train_rl_self_contained.py (the NLA reconstruction-RL)
pretrain/    # pretrain the naive future-lens: FineFineWeb datagen config + finalize_naive_data.py
evals/       # two eval families (below)
interface/   # WeirdChat lens playground (FastAPI: weirdchat_lens.py + weirdchat_ui.html)
configs/     # datagen + RL yaml configs
```

## Stages

**1. Pretrain the naive future-lens** — `pretrain/`
```bash
# datagen: extract L62 activations over a FineFineWeb slice + build (activation → continuation) pairs
python -m nla.datagen.run_pipeline --config configs/datagen/qwen3_8b_finefineweb_100k.yaml
python pretrain/finalize_naive_data.py --labeled <raw.parquet> --meta <raw.parquet.meta.json> --out data/naive_L62
# SFT the future-lens (LoRA on the base model, activation injected at a marker token)
python -m nla.train_sft --mode av --parquet data/naive_L62/train.parquet --sidecar ... --save-dir ckpts/av_L62
```

**2. NLA reconstruction-RL** — lives in `nla/`
```bash
python -m nla.train_rl_self_contained --av-ckpt ckpts/av_L62/iter_XXXX --base-ckpt Qwen/Qwen3.6-27B \
  --rl-parquet data/ae_L62/train.parquet --sidecar ... --save-dir ckpts/rl_ae_L62 \
  --evals base_fve text_judges --eval-every 100    # see configs/rl_*.yaml
```

**3. Evals** — `evals/`
- **A.6 intermediate-surfacing** (does the readout surface the model's hidden intermediate concept):
  `readout_naive.py` (per-variant readouts, raw / Jacobian / mismatch feeds) → `judge_batch.py` or
  `judge_intermediates.py` (Sonnet-5, discrimination = hit − decoy-FP). Variants in
  `naive_variants.json`; the 6 rebuilt A.6 sets in `datasets/`.
- **Fed-layer sweep** (does the lens track the *workspace* (J-lens) or the *answer*, vs read depth):
  `fedlayer_sweep_eval.py` → `judge_fedlayer.py` → `plot_fedlayer.py` / `plot_fedlayer_v2.py`.

**4. Interface** — `interface/`
```bash
BASE_CKPT=Qwen/Qwen3.6-27B JDIR=<jlens dir> HEADS=<horizon heads> \
  NAIVE_CKPT=ckpts/av_L62/iter_XXXX AO_CKPT=... RL_CKPT=... PORT=8801 \
  python interface/weirdchat_lens.py
```
Click any token → read its L62 activation out through {logit-lens, J-lens, fitted head} × {any
source layer}, plus side-by-side future-lens rollouts. Custom user+assistant transcripts supported.

## Requirements / secrets
- `Qwen/Qwen3.6-27B`, a FineFineWeb slice (`m-a-p/FineFineWeb`), 1× H200-class GPU (bf16).
- Keys come from the **environment only** — `HF_TOKEN` (downloads), `ANTHROPIC_API_KEY`
  (Sonnet-5 judges + the interface's future-lens rollouts). Nothing is hard-coded. Interface basic
  auth defaults to `claude`/`claube`; override with `PG_USER`/`PG_PASS`.
