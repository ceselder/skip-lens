"""Full layer x layer cosine-similarity matrix of the residual stream (Qwen3.6-27B), on the
disagreement set, last token. Mean DIRECTION per hidden state (mean of unit vectors). Saves the
RAW matrix and the rogue-dim-corrected (per-dim standardized) matrix. Writes layer_cos_matrix.json.
"""
import glob
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.6-27B"
dev = "cuda"
CTX = 256
EVALS = "/workspace/cnla/skip-lens/evals/datasets_fed"
OUT = "/workspace/cnla/skip-lens/data/meansub/layer_cos_matrix.json"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)
prompts = [it["prompt"] for f in sorted(glob.glob(EVALS + "/lens-eval-*.json"))
           for it in json.load(open(f))["items"]]

sum_unit = s1 = s2 = None
n = 0
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids[:, -CTX:].to(dev)
    hs = model(ids, output_hidden_states=True).hidden_states
    H = torch.stack([h[0, -1].float() for h in hs]).cpu().numpy()   # (NLp1, d)
    if sum_unit is None:
        sum_unit = np.zeros_like(H, dtype=np.float64)
        s1 = np.zeros_like(H, dtype=np.float64); s2 = np.zeros_like(H, dtype=np.float64)
    sum_unit += H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    s1 += H; s2 += H * H
    n += 1
    if n % 100 == 0:
        print(f"[cosmat] {n}/{len(prompts)}", flush=True)

mean_unit = sum_unit / n                              # (NLp1, d) mean direction
mean = s1 / n
std = np.sqrt(np.maximum(s2 / n - mean * mean, 0.0)) + 1e-8
mean_z = mean / std                                   # per-dim standardized mean (rogue-corrected)


def matrix(M):
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    return (Mn @ Mn.T).tolist()


json.dump({"n_items": n, "n_hidden_states": mean_unit.shape[0],
           "note": "index 0 = embeddings; i>=1 = output of decoder layer i-1",
           "cos_raw": matrix(mean_unit), "cos_standardized": matrix(mean_z)},
          open(OUT, "w"))
print("saved", OUT)
