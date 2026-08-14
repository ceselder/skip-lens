"""Per-layer residual-add dynamics on the disagreement set (last token, all layers):
  ||h_L||        mean norm of the running stream
  ||dL||         mean norm of the layer's add  (Δ_L = h_L - h_{L-1})   <- "mean of the norm of the adds"
  growth         mean ||h_L|| / ||h_{L-1}||
  g              mean (Δ_L · ĥ_{L-1}) / ||h_{L-1}||   signed: + amplifies previous content, - shrinks it
                 ("mean shrinkage of previous layers": negative g = shrinkage)
  cos(dL,h_{L-1})  add-vs-stream direction: >0 write along stream (amplify), ~0 orthogonal (new content)
Writes layer_adds.json.
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
OUT = "/workspace/cnla/skip-lens/data/meansub/layer_adds.json"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)
prompts = [it["prompt"] for f in sorted(glob.glob(EVALS + "/lens-eval-*.json"))
           for it in json.load(open(f))["items"]]

hnorm = dnorm = growth = g = dcos = None
n = 0
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids[:, -CTX:].to(dev)
    hs = model(ids, output_hidden_states=True).hidden_states
    H = torch.stack([h[0, -1].float() for h in hs]).cpu().numpy()   # (NLp1, d)
    NLp1 = H.shape[0]
    if hnorm is None:
        hnorm = np.zeros(NLp1); dnorm = np.zeros(NLp1); growth = np.zeros(NLp1)
        g = np.zeros(NLp1); dcos = np.zeros(NLp1)
    hn = np.linalg.norm(H, axis=1)                                  # (NLp1,)
    hnorm += hn
    for i in range(1, NLp1):
        d = H[i] - H[i - 1]
        dn = np.linalg.norm(d) + 1e-8
        h1 = hn[i - 1] + 1e-8
        dnorm[i] += dn
        growth[i] += hn[i] / h1
        g[i] += float(d @ H[i - 1]) / (h1 * h1)                     # (Δ·ĥ_{i-1})/||h_{i-1}||
        dcos[i] += float(d @ H[i - 1]) / (dn * h1)
    n += 1
    if n % 100 == 0:
        print(f"[adds] {n}/{len(prompts)}", flush=True)

hnorm /= n; dnorm /= n; growth /= n; g /= n; dcos /= n
lab = ["emb"] + [str(i) for i in range(NLp1 - 1)]
out = {"n_items": n, "n_hidden_states": NLp1, "labels": lab,
       "h_norm": hnorm.tolist(), "add_norm": dnorm.tolist(), "growth": growth.tolist(),
       "shrinkage_g": g.tolist(), "add_stream_cos": dcos.tolist()}
json.dump(out, open(OUT, "w"), indent=1)

print("\n  L | mean||h|| | mean||dL|| | growth | g(+amp/-shrink) | cos(dL,h-1)")
for i in list(range(0, 6)) + list(range(8, NLp1 - 6, 6)) + list(range(NLp1 - 6, NLp1)):
    l = "emb" if i == 0 else f"L{i-1:>2}"
    gg = "  -  " if i == 0 else f"{g[i]:+.3f}"
    gr = "  -  " if i == 0 else f"{growth[i]:.3f}"
    dc = "  -  " if i == 0 else f"{dcos[i]:+.3f}"
    dn = "  -  " if i == 0 else f"{dnorm[i]:7.1f}"
    print(f"{l:>4}|  {hnorm[i]:7.1f} | {dn} | {gr} |    {gg}      | {dc}")
print("saved", OUT)
