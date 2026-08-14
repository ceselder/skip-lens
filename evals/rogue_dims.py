"""Is the high adjacent-layer cosine a real shared basis, or an artifact of a few outlier
('rogue' / 'massive-activation') dimensions? (Timkey & van Schijndel 2021; Sun et al. 2024.)

Over the disagreement set, per hidden state (last token): per-dim mean/std, the top outlier dims,
and adjacent-layer cosine of the mean DIRECTIONS recomputed with the top-k outlier dims removed and
after per-dim standardization. If cosine stays high after removal -> broad shared basis; if it
collapses -> it was a handful of dims. Writes rogue_dims.json.
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
OUT = "/workspace/cnla/skip-lens/data/meansub/rogue_dims.json"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)
prompts = [it["prompt"] for f in sorted(glob.glob(EVALS + "/lens-eval-*.json"))
           for it in json.load(open(f))["items"]]

s1 = s2 = None
n = 0
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids[:, -CTX:].to(dev)
    hs = model(ids, output_hidden_states=True).hidden_states
    H = torch.stack([h[0, -1].float() for h in hs]).cpu().numpy()   # (NLp1, d)
    if s1 is None:
        s1 = np.zeros_like(H, dtype=np.float64); s2 = np.zeros_like(H, dtype=np.float64)
    s1 += H; s2 += H * H
    n += 1
    if n % 100 == 0:
        print(f"[rogue] {n}/{len(prompts)}", flush=True)

mean = s1 / n                                   # (NLp1, d) per-layer per-dim mean
var = np.maximum(s2 / n - mean * mean, 0.0)
std = np.sqrt(var) + 1e-8
NLp1, d = mean.shape


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def adj_cos(M):
    return [cos(M[i], M[i + 1]) for i in range(NLp1 - 1)]


# variance concentration: what fraction of total per-dim variance is in the top dims
tot_var = var.sum(1)                            # (NLp1,)
def top_var_frac(l, k):
    return float(np.sort(var[l])[::-1][:k].sum() / (tot_var[l] + 1e-8))

# outlier dims by |mean| (massive activations sit at fixed dims across the whole corpus)
def top_mean_dims(l, k):
    idx = np.argsort(np.abs(mean[l]))[::-1][:k]
    return [(int(j), round(float(mean[l][j]), 1)) for j in idx]

# cosine of mean directions with top-k highest-|mean| dims zeroed (union across the two layers)
def adj_cos_drop(k):
    out = []
    for i in range(NLp1 - 1):
        drop = set(np.argsort(np.abs(mean[i]))[::-1][:k]) | set(np.argsort(np.abs(mean[i + 1]))[::-1][:k])
        m = np.ones(d); m[list(drop)] = 0.0
        out.append(cos(mean[i] * m, mean[i + 1] * m))
    return out

# cosine of mean directions after per-dim standardization (whitening the outlier scale)
mean_std = mean / std                            # z-scored mean vector per layer
adj_std = adj_cos(mean_std)

raw = adj_cos(mean)
FED = [63, 62, 55, 48, 42, 34, 26]
diag = {
    "n_items": n, "n_hidden_states": NLp1, "d_model": d,
    "note": "index 0 = embeddings; i>=1 = output of decoder layer i-1; cosines are of per-layer MEAN vectors",
    "adj_cos_raw": raw,
    "adj_cos_drop1": adj_cos_drop(1), "adj_cos_drop3": adj_cos_drop(3), "adj_cos_drop10": adj_cos_drop(10),
    "adj_cos_standardized": adj_std,
    "top_var_frac_top1": [top_var_frac(l, 1) for l in range(NLp1)],
    "top_var_frac_top3": [top_var_frac(l, 3) for l in range(NLp1)],
    "top_var_frac_top10": [top_var_frac(l, 10) for l in range(NLp1)],
    "top_mean_dims": {str(l - 1 if l else "emb"): top_mean_dims(l, 5) for l in [1] + [f + 1 for f in FED]},
}
json.dump(diag, open(OUT, "w"), indent=1)

print(f"\nNL={NLp1-1}; n={n}")
print("\nadjacent-layer cosine of MEAN directions, raw vs top-k-|mean|-dims removed vs standardized:")
print(" transition |  raw  | -top1 | -top3 | -top10 | z-std | top1 var-frac")
for i in range(NLp1 - 1):
    lo = "emb" if i == 0 else f"L{i-1}"
    hi = f"L{i}"
    if i <= 2 or i >= NLp1 - 4 or i % 8 == 0 or (i - 1) in FED:
        print(f" {lo:>3}->{hi:<4}| {raw[i]:.3f} | {adj_cos_drop(1)[i]:.3f} | {adj_cos_drop(3)[i]:.3f} | "
              f"{adj_cos_drop(10)[i]:.3f} | {adj_std[i]:.3f} | {top_var_frac(i,1):.3f}")
print("\ntop-|mean| dims (dim, mean value) at a few layers:")
for l in [1] + [f + 1 for f in FED]:
    print(f"  {'L'+str(l-1):>4}: {top_mean_dims(l,5)}")
print("saved", OUT)
