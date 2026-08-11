"""Quantitative J vs R lens comparison: how well does unembed(M @ h_l) match
the model's own final-layer next-token distribution?

For 16 held-out prompts, at every valid position (skip first 16, drop last):
  * top-1 agreement: lens argmax == model argmax
  * top-8 recall:    model argmax in lens top-8
  * mean KL(model || lens) over the vocab (both at temperature 1)

Lenses: logit-lens (no transport), J (spike control), R (spike), Joff
(official 48-prompt WikiText J), at L42 and L55.

Usage:
  export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4
  /workspace/cnla_venv/bin/python rlens_agreement.py
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

BASE = "Qwen/Qwen3.6-27B"
SRC = [42, 55]
TGT = 62
SPIKE = "/workspace/cnla/results/rlens_spike"
OFFICIAL = "/workspace/cnla/results/jlens"
PARQUET = "/workspace/cnla/skip-lens/data/fl_big/sft_train.parquet"
N_VAL = 16
SKIP_FIRST = 16


def main():
    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder

    mats = {}
    for tag, base in [("J", SPIKE), ("R", SPIKE), ("Joff", OFFICIAL)]:
        pre = "J" if tag == "Joff" else tag
        mats[tag] = {
            l: torch.from_numpy(
                np.load(f"{base}/{pre}_L{l}_to_L{TGT}.npy")
            ).float().cuda()
            for l in SRC
        }

    df = pd.read_parquet(PARQUET, columns=["ctx_text"])
    texts = df["ctx_text"].iloc[100000:].drop_duplicates()
    val_prompts = [t for t in texts if isinstance(t, str) and len(t) >= 300][:N_VAL]

    tok = AutoTokenizer.from_pretrained(BASE)
    hf = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    model = from_hf(hf, tok)
    final = model.n_layers - 1

    lens_names = [f"{t}_L{l}" for l in SRC for t in ("logit", "J", "R", "Joff")]
    agree1 = {k: 0 for k in lens_names}
    recall8 = {k: 0 for k in lens_names}
    kl_sum = {k: 0.0 for k in lens_names}
    n_pos = 0

    for prompt in val_prompts:
        input_ids = model.encode(prompt, max_length=128)
        seq_len = input_ids.shape[1]
        if seq_len <= SKIP_FIRST + 1:
            continue
        with torch.no_grad(), ActivationRecorder(model.layers, at=[*SRC, final]) as rec:
            model.forward(input_ids)
            acts = {l: rec.activations[l][0].detach().float() for l in rec.activations}
        pos = torch.arange(SKIP_FIRST, seq_len - 1)
        with torch.no_grad():
            model_logits = model.unembed(acts[final][pos]).float()
            model_logp = F.log_softmax(model_logits, dim=-1)
            model_p = model_logp.exp()
            model_top1 = model_logits.argmax(-1)
            for l in SRC:
                h = acts[l][pos]  # [P, d]
                for t in ("logit", "J", "R", "Joff"):
                    x = h if t == "logit" else h @ mats[t][l].T
                    logits = model.unembed(x).float()
                    logp = F.log_softmax(logits, dim=-1)
                    key = f"{t}_L{l}"
                    top1 = logits.argmax(-1)
                    agree1[key] += (top1 == model_top1).sum().item()
                    top8 = logits.topk(8, dim=-1).indices
                    recall8[key] += (top8 == model_top1[:, None]).any(-1).sum().item()
                    kl_sum[key] += (model_p * (model_logp - logp)).sum(-1).sum().item()
        n_pos += len(pos)

    print(f"n_prompts={len(val_prompts)} n_positions={n_pos}")
    out = {"n_positions": n_pos, "lenses": {}}
    for k in lens_names:
        row = {
            "top1_agree": agree1[k] / n_pos,
            "top8_recall_of_model_top1": recall8[k] / n_pos,
            "mean_KL_model_to_lens": kl_sum[k] / n_pos,
        }
        out["lenses"][k] = row
        print(f"{k:10s}: top1={row['top1_agree']:.3f}  "
              f"top8recall={row['top8_recall_of_model_top1']:.3f}  "
              f"KL={row['mean_KL_model_to_lens']:.3f}")
    with open(f"{SPIKE}/agreement.json", "w") as f:
        json.dump(out, f, indent=1)
    print("=== RLENS AGREEMENT DONE ===", flush=True)


if __name__ == "__main__":
    main()
