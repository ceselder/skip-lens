"""Residual-stream geometry across ALL layers, on the disagreement set.

For each hidden state (embeddings + every decoder-layer output) at the last token:
  * mean DIRECTION = mean of per-token unit vectors (normalize-then-mean; magnitude-free).
  * cosine similarity between layer mean-directions (adjacent, to-emb, to-last, to-mid, full matrix).
  * "how much gets added" per layer L: delta_L = h_L - h_{L-1} (the layer's attn+MLP contribution):
      relative update ||delta_L|| / ||h_L||   and   rotation cos(h_{L-1}, h_L).
Saves diagnostics json + mean_test_dir_by_layer.npz (normed mean dirs for the fed layers, for the
mean-subtraction eval done the RIGHT way: subtract the mean direction, not the raw magnitude-mean).
"""
import glob
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.6-27B"
dev = "cuda"
CTX = 256
FED = [62, 55, 48, 42, 34, 26]
EVALS = "/workspace/cnla/skip-lens/evals/datasets_fed"
OUT = "/workspace/cnla/skip-lens/data/meansub"

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)

prompts = [it["prompt"] for f in sorted(glob.glob(EVALS + "/lens-eval-*.json"))
           for it in json.load(open(f))["items"]]

sum_raw = sum_unit = None
add_rel = add_norm = rot = None
n = 0
for p in prompts:
    ids = tok(p, return_tensors="pt").input_ids[:, -CTX:].to(dev)
    hs = model(ids, output_hidden_states=True).hidden_states       # tuple len NL+1 (0 = embeddings)
    H = torch.stack([h[0, -1].float() for h in hs]).cpu().numpy()  # (NL+1, d)
    if sum_raw is None:
        NLp1, d = H.shape
        sum_raw = np.zeros((NLp1, d)); sum_unit = np.zeros((NLp1, d))
        add_rel = np.zeros(NLp1); add_norm = np.zeros(NLp1); rot = np.zeros(NLp1)
    nrm = np.linalg.norm(H, axis=1, keepdims=True) + 1e-8
    sum_raw += H
    sum_unit += H / nrm
    dif = H[1:] - H[:-1]                                            # (NL, d): contribution of decoder layer i-1
    dn = np.linalg.norm(dif, axis=1)
    add_norm[1:] += dn
    add_rel[1:] += dn / (np.linalg.norm(H[1:], axis=1) + 1e-8)
    add_rel[0] = np.nan
    cc = (H[1:] * H[:-1]).sum(1) / (np.linalg.norm(H[1:], axis=1) * np.linalg.norm(H[:-1], axis=1) + 1e-8)
    rot[1:] += cc
    n += 1
    if n % 100 == 0:
        print(f"[geom] {n}/{len(prompts)}", flush=True)

mean_raw = sum_raw / n
mean_unit = sum_unit / n
add_rel /= n; add_norm /= n; rot /= n
NLp1 = mean_raw.shape[0]
NL = NLp1 - 1                                                       # number of decoder layers


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# --- save normed mean directions for the fed layers (decoder layer l = hidden_states[l+1]) ---
dirs = {str(l): mean_unit[l + 1].astype(np.float32) for l in FED}
np.savez(f"{OUT}/mean_test_dir_by_layer.npz", **dirs)

# --- cosine profiles over the mean DIRECTIONS (index 0 = emb, i>=1 = decoder layer i-1) ---
mid = NLp1 // 2
adj = [cos(mean_unit[i], mean_unit[i + 1]) for i in range(NLp1 - 1)]
to_emb = [cos(mean_unit[i], mean_unit[0]) for i in range(NLp1)]
to_last = [cos(mean_unit[i], mean_unit[-1]) for i in range(NLp1)]
to_mid = [cos(mean_unit[i], mean_unit[mid]) for i in range(NLp1)]
# coarse full matrix at stride 8 (+ always include last)
grid = sorted(set(list(range(0, NLp1, 8)) + [NLp1 - 1]))
mat = [[cos(mean_unit[i], mean_unit[jj]) for jj in grid] for i in grid]

diag = {
    "n_items": n, "n_hidden_states": NLp1, "n_decoder_layers": NL,
    "note": "index 0 = embeddings; index i>=1 = output of decoder layer i-1",
    "mean_raw_norm": [float(np.linalg.norm(mean_raw[i])) for i in range(NLp1)],
    "mean_dir_concentration": [float(np.linalg.norm(mean_unit[i])) for i in range(NLp1)],
    "adjacent_cos": adj, "cos_to_emb": to_emb, "cos_to_last": to_last, "cos_to_mid": to_mid,
    "mid_index": mid,
    "add_rel": [None if np.isnan(x) else float(x) for x in add_rel],
    "add_norm": [float(x) for x in add_norm], "rot_cos": [float(x) for x in rot],
    "grid": grid, "cos_matrix_stride8": mat,
}
json.dump(diag, open(f"{OUT}/layer_geometry.json", "w"), indent=1)

print(f"\nNL={NL} decoder layers (hidden_states={NLp1}); n={n} items")
print("\n L | dir-concentration | adj-cos(L,L+1) | rel-add ||Δ||/||h|| | rot cos(h-1,h) | ||mean_raw||")
def row(i):
    lbl = "emb" if i == 0 else f"L{i-1:>2}"
    a = f"{adj[i]:.3f}" if i < len(adj) else "  -  "
    r = "  -  " if (i == 0 or np.isnan(add_rel[i])) else f"{add_rel[i]:.3f}"
    rc = "  -  " if i == 0 else f"{rot[i]:.3f}"
    print(f"{lbl:>4}|   {np.linalg.norm(mean_unit[i]):.3f}        |    {a}      |     {r}        |    {rc}    | {np.linalg.norm(mean_raw[i]):7.1f}")
for i in list(range(0, 6)) + list(range(8, NLp1 - 6, 6)) + list(range(NLp1 - 6, NLp1)):
    row(i)
print("\ncos(emb, last decoder layer) =", round(cos(mean_unit[0], mean_unit[-1]), 3))
print("cos(L42dir, L62dir) =", round(cos(mean_unit[43], mean_unit[63]), 3),
      "| cos(L26dir, L42dir) =", round(cos(mean_unit[27], mean_unit[43]), 3))
print("saved layer_geometry.json + mean_test_dir_by_layer.npz")
