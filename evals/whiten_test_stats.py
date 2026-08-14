"""Whitening ablation, TEST side. Per-layer ZCA whitening stats (mu, Sigma^{-1/2}) estimated over the
fl_big distribution at each fed layer (forward the first N ctx_texts, grab last token). Full-rank
(N >> d) so the covariance is estimable; shrinkage alpha=0.15. Saves whiten_test_stats.npz keyed by
layer. At eval, feed W_l (h_l - mu_l) for the disagreement items.
"""
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.6-27B"
dev = "cuda"
CTX = 256
LAYERS = [62, 55, 48, 42, 34, 26]
N = 12000
ALPHA = 0.15
SRC = "/workspace/cnla/skip-lens/data/fl_big/av_L62_150k.parquet"
OUT = "/workspace/cnla/skip-lens/data/meansub/whiten_test_stats.npz"

pf = pq.ParquetFile(SRC)
parts, have = [], 0
for rg in range(pf.num_row_groups):
    parts.append(pf.read_row_group(rg))
    have += parts[-1].num_rows
    if have >= N:
        break
ctx = pa.concat_tables(parts)["ctx_text"].to_pylist()[:N]

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)
grab = {}
for l in LAYERS:
    model.model.layers[l].register_forward_hook(
        lambda m, i, o, L=l: grab.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))

X = {l: np.zeros((N, 5120), dtype=np.float32) for l in LAYERS}
k = 0
for t in ctx:
    ids = tok(t, return_tensors="pt").input_ids[:, -CTX:].to(dev)
    if ids.shape[1] == 0:
        continue
    model(ids)
    for l in LAYERS:
        X[l][k] = grab[l][0, -1].float().cpu().numpy()
    k += 1
    if k % 1000 == 0:
        print(f"[whiten-stats] {k}/{N}", flush=True)

stats = {}
for l in LAYERS:
    Xl = X[l][:k]
    mu = Xl.mean(0)
    Xc = Xl - mu
    S = (Xc.T @ Xc) / len(Xc)
    S = (1 - ALPHA) * S + ALPHA * (np.trace(S) / S.shape[0]) * np.eye(S.shape[0], dtype=S.dtype)
    w, U = np.linalg.eigh(S.astype(np.float64))
    W = ((U * (w ** -0.5)) @ U.T).astype(np.float32)
    stats[f"mu_{l}"] = mu.astype(np.float32)
    stats[f"W_{l}"] = W
    print(f"L{l}: cond={w.max()/w.min():.1f} ||mu||={np.linalg.norm(mu):.1f}", flush=True)

np.savez(OUT, **stats)
print("saved", OUT, "over", k, "items")
