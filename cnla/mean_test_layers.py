"""Per-layer mean activation over the TEST distribution (the disagreement eval set).

Base model (no LoRA), forward each disagreement prompt, grab the residual at the last real token
for each fed layer, average over items. Written the same way the fed-layer eval grabs h[l]
(last-256-token context, position -1) so the subtracted mean matches the fed activation exactly.
Writes mean_test_by_layer.npz keyed by layer index.
"""
import glob
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.6-27B"
dev = "cuda"
CTX = 256
LAYERS = [62, 55, 48, 42, 34, 26]
EVALS = "/workspace/cnla/skip-lens/evals/datasets_fed"
OUT = "/workspace/cnla/skip-lens/data/meansub/mean_test_by_layer.npz"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)

grab = {}
for l in LAYERS:
    model.model.layers[l].register_forward_hook(
        lambda m, i, o, L=l: grab.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))

prompts = []
for f in sorted(glob.glob(EVALS + "/lens-eval-*.json")):
    for it in json.load(open(f))["items"]:
        prompts.append(it["prompt"])

acc = {l: np.zeros(5120, dtype=np.float64) for l in LAYERS}
n = 0
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids[:, -CTX:].to(dev)
    model(ids)
    for l in LAYERS:
        acc[l] += grab[l][0, -1].float().cpu().numpy()
    n += 1
    if n % 100 == 0:
        print(f"[mean-test] {n}/{len(prompts)}", flush=True)

means = {str(l): (acc[l] / n).astype(np.float32) for l in LAYERS}
np.savez(OUT, **means)
print(f"test means over {n} items -> {OUT}")
print("norms", {l: round(float(np.linalg.norm(means[str(l)])), 1) for l in LAYERS})
