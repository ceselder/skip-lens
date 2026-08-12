"""AUDIT (GPU, main venv): consistency with the ORIGINAL pooled estimator.

jlens.fitting.jacobian_for_prompt (the paper's reference reduction) computes
J_pooled = mean_t [ sum_{t' in valid, t' >= t} dh62[t']/dh42[t] ]
         ~= sum_{delta=0..~seq} Jbar^(delta)   (up to position-edge weighting).
If the comb-estimated family is right, cos(J_pooled_ref, cumsum_k Jbar) must
increase monotonically in k and clearly exceed cos(J_pooled_ref, Jbar^0).

Fits the reference on N_REF held-out prompts at the same seq len (512).

Run:  CUDA_VISIBLE_DEVICES=2 /workspace/venv/bin/python diag_offsets_pooledref.py
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import configure_logging
from jlens.fitting import jacobian_for_prompt
from jlens.hf import from_hf

OUT = "/workspace/results/offset_jlens"
AUD = "/workspace/results/offset_jlens_audit"
BASE = "Qwen/Qwen3.6-27B"
SRC, TGT = 42, 62
SEQ = 512
N_REF = int(os.environ.get("N_REF", "6"))
DIM_BATCH = int(os.environ.get("DIM_BATCH", "16"))

configure_logging()


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    hf = (AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        .cuda().eval())
    model = from_hf(hf, tok)
    prompts = json.load(open(f"{AUD}/heldout_prompts.json"))[:N_REF]

    acc = None
    for pi, p in enumerate(prompts):
        J, seq_len, n_valid = jacobian_for_prompt(
            model, p, [SRC], target_layer=TGT,
            dim_batch=DIM_BATCH, max_seq_len=SEQ)
        acc = J[SRC] if acc is None else acc + J[SRC]
        print(f"ref prompt {pi+1}/{N_REF} seq={seq_len} n_valid={n_valid} "
              f"|J|={J[SRC].norm():.2f}", flush=True)
    Jref = (acc / N_REF).double()
    torch.save(Jref.float(), f"{AUD}/pooled_ref_{N_REF}prompts.pt")

    Js = [torch.from_numpy(np.load(f"{OUT}/Jbar_L{SRC}_to_L{TGT}_off{d}.npy"))
          .double() for d in range(16)]

    def cos(a, b):
        a, b = a.flatten(), b.flatten()
        return float(a @ b / (a.norm() * b.norm()))

    print(f"\n||Jref({N_REF} prompts)|| = {Jref.norm():.3f}")
    run = torch.zeros_like(Js[0])
    out = []
    for k in range(16):
        run += Js[k]
        c = cos(Jref, run)
        out.append({"k": k, "cos_ref_cumsum": c,
                    "fro_cumsum": float(run.norm())})
        print(f"cos(Jref, sum_0^{k:>2} Jbar) = {c:.4f}   |sum|={run.norm():.2f}")
    print(f"cos(Jref, Jbar^0 only)   = {cos(Jref, Js[0]):.4f}")
    print(f"cos(Jref, offpooled/mean)= {cos(Jref, run / 16):.4f} (same as sum)")
    json.dump({"n_ref": N_REF, "fro_ref": float(Jref.norm()), "rows": out},
              open(f"{AUD}/pooledref_audit.json", "w"), indent=2)
    print(f"wrote {AUD}/pooledref_audit.json")


if __name__ == "__main__":
    main()
