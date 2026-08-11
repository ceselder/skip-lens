"""Compare J-lens vs R-lens readouts on Qwen3.6-27B.

Loads the spike-fit J control and R matrices (plus the official 48-prompt
WikiText J for reference), runs a handful of validation prompts, and prints
side-by-side top-8 token readouts of unembed(M @ h_l) at a few positions.

Usage (on the GPU box):
  export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4
  /workspace/cnla_venv/bin/python rlens_readout.py
"""

import json
import os

import numpy as np
import torch

BASE = "Qwen/Qwen3.6-27B"
SRC = [42, 55]
TGT = 62
SPIKE = "/workspace/cnla/results/rlens_spike"
OFFICIAL = "/workspace/cnla/results/jlens"
PARQUET = "/workspace/cnla/skip-lens/data/fl_big/sft_train.parquet"
TOPK = 8

HANDWRITTEN = [
    "The Eiffel Tower is located in the city of Paris, which is the capital of "
    "France. The tower was built in 1889 for the World's Fair and has since "
    "become one of the most recognizable landmarks in the world. Every year, "
    "millions of tourists visit the tower to enjoy the view from",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return "
    "fibonacci(n - 1) + fibonacci(n - 2)\n\n\ndef main():\n    # print the "
    "first ten Fibonacci numbers\n    for i in range(10):\n        print(",
]


def load_matrices():
    mats = {}
    for tag, path in [
        ("J", f"{SPIKE}/J_L{{l}}_to_L{TGT}.npy"),
        ("R", f"{SPIKE}/R_L{{l}}_to_L{TGT}.npy"),
        ("Joff", f"{OFFICIAL}/J_L{{l}}_to_L{TGT}.npy"),
    ]:
        mats[tag] = {}
        for l in SRC:
            p = path.format(l=l)
            if os.path.exists(p):
                mats[tag][l] = torch.from_numpy(np.load(p)).float()
    return mats


def main():
    mats = load_matrices()
    d = 5120
    stats = {}
    print("=== matrix stats ===", flush=True)
    for l in SRC:
        row = {}
        for tag in ("J", "R", "Joff"):
            if l in mats.get(tag, {}):
                M = mats[tag][l]
                row[f"{tag}_norm_over_sqrtd"] = M.norm().item() / d**0.5
                row[f"{tag}_finite"] = bool(torch.isfinite(M).all())
        for a, b, name in [("R", "J", "cos_R_J"), ("J", "Joff", "cos_J_Joff"),
                           ("R", "Joff", "cos_R_Joff")]:
            if l in mats.get(a, {}) and l in mats.get(b, {}):
                x, y = mats[a][l].flatten(), mats[b][l].flatten()
                row[name] = torch.nn.functional.cosine_similarity(x, y, dim=0).item()
        stats[l] = row
        print(f"L{l}: " + "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()), flush=True)

    import pandas as pd
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf
    from jlens.hooks import ActivationRecorder

    df = pd.read_parquet(PARQUET, columns=["ctx_text"])
    texts = df["ctx_text"].iloc[100000:].drop_duplicates()
    val_prompts = [t for t in texts if isinstance(t, str) and len(t) >= 300][:4]
    val_prompts += HANDWRITTEN

    tok = AutoTokenizer.from_pretrained(BASE)
    hf = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    model = from_hf(hf, tok)

    def topk_str(logits, k=TOPK):
        vals, idx = logits.topk(k)
        return " | ".join(tok.decode([i]).replace("\n", "\\n") for i in idx.tolist())

    results = []
    for pi, prompt in enumerate(val_prompts):
        input_ids = model.encode(prompt, max_length=128)
        seq_len = input_ids.shape[1]
        positions = sorted({seq_len // 2, seq_len - 1})
        with torch.no_grad(), ActivationRecorder(model.layers, at=[*SRC, model.n_layers - 1]) as rec:
            model.forward(input_ids)
            acts = {l: rec.activations[l][0].detach().float() for l in rec.activations}

        print(f"\n=== prompt {pi}: ...{prompt[-80:]!r}", flush=True)
        entry = {"prompt": prompt, "positions": []}
        for pos in positions:
            ctx_tail = tok.decode(input_ids[0, max(0, pos - 11): pos + 1].tolist())
            with torch.no_grad():
                model_logits = model.unembed(acts[model.n_layers - 1][pos]).float().cpu()
            actual_next = (
                tok.decode([input_ids[0, pos + 1].item()]) if pos + 1 < seq_len else None
            )
            pos_entry = {
                "pos": pos,
                "ctx_tail": ctx_tail,
                "model_top": topk_str(model_logits, 3),
                "actual_next": actual_next,
                "layers": {},
            }
            print(f"\n-- pos {pos} | ctx: ...{ctx_tail!r}", flush=True)
            print(f"   model final top-3 : {topk_str(model_logits, 3)}   "
                  f"(actual next: {actual_next!r})", flush=True)
            for l in SRC:
                h = acts[l][pos]
                lay = {}
                with torch.no_grad():
                    naive = model.unembed(h).float().cpu()
                lay["logitlens"] = topk_str(naive)
                print(f"   L{l} logit-lens    : {lay['logitlens']}", flush=True)
                for tag in ("J", "R", "Joff"):
                    if l in mats.get(tag, {}):
                        with torch.no_grad():
                            transported = mats[tag][l].to(h.device) @ h
                            logits = model.unembed(transported).float().cpu()
                        lay[tag] = topk_str(logits)
                        print(f"   L{l} {tag:<4}-lens     : {lay[tag]}", flush=True)
                pos_entry["layers"][l] = lay
            entry["positions"].append(pos_entry)
        results.append(entry)

    with open(f"{SPIKE}/readouts.json", "w") as f:
        json.dump({"stats": {str(k): v for k, v in stats.items()},
                   "readouts": results}, f, indent=1)
    print(f"\nsaved {SPIKE}/readouts.json", flush=True)
    print("=== RLENS READOUT DONE ===", flush=True)


if __name__ == "__main__":
    main()
