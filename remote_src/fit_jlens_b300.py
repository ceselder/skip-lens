"""Refit the J-lens for Qwen3.6-27B on the B300 using Anthropic's OFFICIAL
jlens.fit (github.com/anthropics/jacobian-lens). Adapted from the CA-MTL-4
script fit_jlens_official.py: B300 paths, SRC layers for the cnla project,
no sanity-compare block. Saves J_L{l}_to_L62.npy ([5120,5120] float32) to
/workspace/cnla/results/jlens/.

Env knobs:
  N_PROMPTS  (default 48)  - slice of the WikiText prompt list
  DIM_BATCH  (default 64)  - jacobian rows per backward pass
  CKPT_NAME  (default fit_ckpt_b300.pt)
"""
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens.hf import from_hf
from jlens.fitting import fit

BASE = "Qwen/Qwen3.6-27B"
R = "/workspace/cnla"
SRC = [10, 18, 26, 34, 42, 48, 55]
TGT = 62                              # block-62 output = penultimate
OUT = f"{R}/results/jlens"
N_PROMPTS = int(os.environ.get("N_PROMPTS", "48"))
DIM_BATCH = int(os.environ.get("DIM_BATCH", "64"))
CKPT = f"{OUT}/" + os.environ.get("CKPT_NAME", "fit_ckpt_b300.pt")
os.makedirs(OUT, exist_ok=True)

prompts = json.load(open(f"{R}/data/wikitext_prompts.json"))[:N_PROMPTS]
print(f"[fit] {len(prompts)} WikiText prompts | src={SRC} tgt=L{TGT} "
      f"dim_batch={DIM_BATCH} ckpt={CKPT}", flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
hf = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
model = from_hf(hf, tok)
print(f"[fit] LensModel: n_layers={model.n_layers} d_model={model.d_model}", flush=True)

fit(model, prompts, source_layers=SRC, target_layer=TGT, dim_batch=DIM_BATCH,
    max_seq_len=128, checkpoint_path=CKPT, checkpoint_every=4, resume=True)

# extract running mean J = jacobian_sum / n_done from the checkpoint, save our format
st = torch.load(CKPT, map_location="cpu", weights_only=True)
Jsum, n = st["jacobian_sum"], st["n_done"]
print(f"[fit] averaged over n_done={n} prompts", flush=True)
for l in SRC:
    J = (Jsum[l] / n).float().numpy().astype("float32")
    np.save(f"{OUT}/J_L{l}_to_L{TGT}.npy", J)
    print(f"  saved J_L{l}_to_L{TGT}  shape={J.shape} "
          f"|J|_F={float(np.linalg.norm(J)):.2f} finite={bool(np.isfinite(J).all())}",
          flush=True)
print("=== B300 J FIT DONE ===", flush=True)
