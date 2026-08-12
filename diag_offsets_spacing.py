"""AUDIT (GPU, main venv): comb-spacing / aliasing contamination check.

The shipped Jbar^(delta) were fitted with comb_spacing=32 and n_offsets=16:
the gradient read at tooth-delta also contains contamination from teeth at
offsets delta+32k, suppressed only by the Rademacher sign trick. If that
machinery is broken (bias, not variance), a refit with a much wider comb
(spacing 128 -> contamination offsets delta+128k, ~zero norm) would deviate
from a same-size spacing-32 refit systematically.

We fit BOTH spacings fresh on held-out prompts via jlens.offset_fitting
.fit_offsets and compare each, per delta, against the shipped matrices.
PASS = cos(fit128, shipped) matches cos(fit32, shipped) up to small-sample
noise for every delta.

Run:  CUDA_VISIBLE_DEVICES=1 /workspace/venv/bin/python diag_offsets_spacing.py
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens import configure_logging
from jlens.hf import from_hf
from jlens.offset_fitting import fit_offsets

OUT = "/workspace/results/offset_jlens"
AUD = "/workspace/results/offset_jlens_audit"
BASE = "Qwen/Qwen3.6-27B"
SRC, TGT = 42, 62
N_OFFSETS = 16
SEQ = 512
DIM_BATCH = int(os.environ.get("DIM_BATCH", "16"))
N32 = int(os.environ.get("N32", "16"))     # prompts for the spacing-32 refit
N128 = int(os.environ.get("N128", "24"))   # more prompts: 4 teeth/prompt vs 15

configure_logging()


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    hf = (AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        .cuda().eval())
    model = from_hf(hf, tok)
    prompts = json.load(open(f"{AUD}/heldout_prompts.json"))

    fits = {}
    for spacing, n in ((32, N32), (128, N128)):
        ckpt = f"{AUD}/refit_spacing{spacing}.pt"
        print(f"=== refit spacing={spacing} on {n} held-out prompts "
              f"(dim_batch={DIM_BATCH}) ===", flush=True)
        J = fit_offsets(
            model, prompts[:n],
            source_layers=[SRC], target_layer=TGT,
            n_offsets=N_OFFSETS, comb_spacing=spacing,
            dim_batch=DIM_BATCH, max_seq_len=SEQ,
            checkpoint_path=ckpt, checkpoint_every=2, resume=True,
        )[SRC]  # [16, d, d]
        fits[spacing] = J

    ship = torch.stack([
        torch.from_numpy(np.load(f"{OUT}/Jbar_L{SRC}_to_L{TGT}_off{d}.npy"))
        for d in range(N_OFFSETS)])

    def cos(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        return float(a @ b / (a.norm() * b.norm()))

    print(f"\n{'d':>2} {'cos(32,ship)':>13} {'cos(128,ship)':>14} "
          f"{'cos(32,128)':>12} {'fro32':>8} {'fro128':>8} {'froship':>8}")
    rows = []
    for d in range(N_OFFSETS):
        c32 = cos(fits[32][d], ship[d])
        c128 = cos(fits[128][d], ship[d])
        cx = cos(fits[32][d], fits[128][d])
        f32, f128, fs = (float(fits[32][d].norm()), float(fits[128][d].norm()),
                         float(ship[d].norm()))
        rows.append({"delta": d, "cos32_ship": c32, "cos128_ship": c128,
                     "cos32_128": cx, "fro32": f32, "fro128": f128,
                     "fro_ship": fs})
        print(f"{d:>2} {c32:>13.4f} {c128:>14.4f} {cx:>12.4f} "
              f"{f32:>8.3f} {f128:>8.3f} {fs:>8.3f}")

    json.dump({"dim_batch": DIM_BATCH, "n32": N32, "n128": N128, "rows": rows},
              open(f"{AUD}/spacing_audit.json", "w"), indent=2)
    print(f"\nwrote {AUD}/spacing_audit.json")


if __name__ == "__main__":
    main()
