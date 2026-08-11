"""Where does fl_big's stored activation actually sit? (position-convention check)

fl_big kept ctx_text. Recompute the model's real L62 residual across ctx_text and
find which position best matches the STORED activation_vector. If it's the last ctx
token (offset 0 from end), fl_big = "activation at last-context-token, response is the
continuation" (standard). A consistent non-zero offset (or an interior match) would be
the collection-convention change vs ae_L62. Also checks response really follows ctx_text.
"""
import numpy as np, torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter

dev = "cuda"; BASE = "Qwen/Qwen3.6-27B"; M = 48
LAYERS = [59, 60, 61, 62, 63]         # sweep around the nominal L62 to catch off-by-one
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True); tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to(dev).eval()
grab = {}
for Lh in LAYERS:
    model.model.layers[Lh].register_forward_hook(
        lambda m, i, o, LL=Lh: grab.__setitem__(LL, (o[0] if isinstance(o, tuple) else o).detach()))

t = pq.ParquetFile("/workspace/cnla/skip-lens/data/fl_big/sft_train.parquet").read_row_group(0)
ctx = t.column("ctx_text").to_pylist()
resp = t.column("response").to_pylist()
ac = t.column("activation_vector").combine_chunks()
acts = ac.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(ac), -1)

# per-layer: cos(stored, layer_output @ last ctx token); and global best (layer, offset-from-end)
cos_last = {L: [] for L in LAYERS}
best_layer, best_off, best_cosv = [], [], []
for i in range(M):
    a = torch.tensor(acts[i], device=dev).float()
    ids = tok(ctx[i], return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev)
    with torch.no_grad():
        model(input_ids=ids)
    bi = (-1.0, None, None)
    for L in LAYERS:
        h = grab[L][0].float()                     # [T, d]
        cos_last[L].append(F.cosine_similarity(a, h[-1], dim=0).item())
        cs = F.cosine_similarity(a.unsqueeze(0), h, dim=1)
        j = int(cs.argmax()); v = float(cs[j])
        if v > bi[0]:
            bi = (v, L, h.shape[0] - 1 - j)         # (cos, layer, offset-from-end)
    best_cosv.append(bi[0]); best_layer.append(bi[1]); best_off.append(bi[2])

print(f"[fl_big position/layer check, M={M}]")
for L in LAYERS:
    print(f"  mean cos(stored, layers[{L}]-output @ LAST ctx token) = {np.mean(cos_last[L]):.3f}")
print(f"  GLOBAL best-match layer  histogram = {dict(sorted(Counter(best_layer).items()))}")
print(f"  GLOBAL best-match offset histogram = {dict(sorted(Counter(best_off).items()))}  (0=last ctx token)")
print(f"  mean best cos = {np.mean(best_cosv):.3f}")
print(f"  ctx sample[0] tail: {ctx[0][-60:]!r}")
print(f"  resp sample[0]:     {resp[0][:60]!r}")
