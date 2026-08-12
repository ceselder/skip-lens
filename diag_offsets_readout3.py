"""AUDIT follow-up 2 (GPU, main venv): affine-offset decomposition of the
J-lens readout deficit, plus readout of the freshly-fitted PAPER-convention
pooled reference J.

An averaged linear transport discards the affine part: the best linear-
plus-constant predictor of h62 from h42 is J h42 + b with
b = E[h62] - J E[h42].  We estimate b on 12 held-out prompts (means over
positions) and evaluate on the other 12, for each transport:
    J0(+b), sum16(+b), Jref(+b)  [paper pooled estimator, 6-prompt refit],
    identity(+b) (= logit lens / its mean-shifted variant), b alone,
    h62true (oracle).
If +b restores J-lens > logit-lens, the matrices are valid and the deficit
is the (expected) missing affine term, not an estimator bug.

Run:  CUDA_VISIBLE_DEVICES=2 /workspace/venv/bin/python diag_offsets_readout3.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/skip-lens")
from transformers import AutoModelForCausalLM, AutoTokenizer

from interface.common import norm_gain, resolve_text_model
from jlens.hf import from_hf

OUT = "/workspace/results/offset_jlens"
AUD = "/workspace/results/offset_jlens_audit"
BASE = "Qwen/Qwen3.6-27B"
SRC, TGT = 42, 62
SEQ = 512
T_LO, STRIDE = 31, 3


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    hf = (AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        .cuda().eval())
    model = from_hf(hf, tok)
    tm = resolve_text_model(hf)
    gain = norm_gain(hf).cuda()
    eps = float(getattr(hf.config.get_text_config(), "rms_norm_eps", 1e-6))
    W_U = hf.lm_head.weight.detach().float().cuda()

    def decode(x):
        u = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        return (u * gain) @ W_U.T

    Jall = torch.stack([
        torch.from_numpy(np.load(f"{OUT}/Jbar_L42_to_L62_off{d}.npy"))
        for d in range(16)]).cuda()
    mats = {"J0": Jall[0].clone(), "sum16": Jall.sum(0), "ident": None}
    del Jall
    ref_path = f"{AUD}/pooled_ref_6prompts.pt"
    if os.path.exists(ref_path):
        mats["Jref"] = torch.load(ref_path, map_location="cpu").float().cuda()

    prompts = json.load(open(f"{AUD}/heldout_prompts.json"))
    stat_prompts, eval_prompts = prompts[:12], prompts[12:24]

    store = {}
    def cap(idx):
        def hook(m, i, o):
            store[idx] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook
    hooks = [tm.layers[SRC].register_forward_hook(cap(SRC)),
             tm.layers[TGT].register_forward_hook(cap(TGT))]

    def get_acts(prompt):
        ids = model.encode(prompt, max_length=SEQ)
        with torch.no_grad():
            out = hf(input_ids=ids, use_cache=False)
        seq_len = ids.shape[1]
        ts = torch.arange(T_LO, seq_len - 2, STRIDE, device="cuda")
        return (ids[0][ts + 1], out.logits[0].float()[ts].argmax(-1),
                store[SRC][0].float()[ts], store[TGT][0].float()[ts])

    try:
        # pass 1: estimate offsets b on stat prompts
        sums = {c: 0.0 for c in mats}
        s62, n_stat = 0.0, 0
        for p in stat_prompts:
            _, _, h42_t, h62_t = get_acts(p)
            for c, m in mats.items():
                pred = h42_t if m is None else h42_t @ m.T
                sums[c] = sums[c] + pred.sum(0).double()
            s62 = s62 + h62_t.sum(0).double()
            n_stat += len(h42_t)
        b = {c: ((s62 - sums[c]) / n_stat).float() for c in mats}
        print(f"stat pass done: n={n_stat}; |b| " +
              " ".join(f"{c}:{b[c].norm():.1f}" for c in mats), flush=True)

        # pass 2: evaluate on eval prompts
        conds = ([f"{c}" for c in mats] + [f"{c}+b" for c in mats]
                 + ["b_only", "h62true"])
        acc = {c: {"t1m": 0.0, "t5m": 0.0, "t1c": 0.0, "lp": 0.0, "cos62": 0.0}
               for c in conds}
        n_pos = 0
        for pi, p in enumerate(eval_prompts):
            nxt, model_next, h42_t, h62_t = get_acts(p)
            rows = {}
            for c, m in mats.items():
                pred = h42_t if m is None else h42_t @ m.T
                rows[c] = pred
                rows[f"{c}+b"] = pred + b[c]
            rows["b_only"] = b["sum16"].expand_as(h42_t)
            rows["h62true"] = h62_t
            for c, x in rows.items():
                lg = decode(x)
                lsm = lg - torch.logsumexp(lg, -1, keepdim=True)
                am = lg.argmax(-1)
                t5 = lg.topk(5, -1).indices
                acc[c]["t1m"] += (am == model_next).double().sum().item()
                acc[c]["t5m"] += (t5 == model_next[:, None]).any(-1).double().sum().item()
                acc[c]["t1c"] += (am == nxt).double().sum().item()
                acc[c]["lp"] += lsm.gather(1, nxt[:, None]).double().sum().item()
                acc[c]["cos62"] += F.cosine_similarity(x, h62_t, -1).double().sum().item()
                del lg, lsm
            n_pos += len(h42_t)
            print(f"eval prompt {pi+1}/12 total={n_pos}", flush=True)
    finally:
        for h in hooks:
            h.remove()

    print(f"\n#### readout3: affine decomposition (eval n={n_pos})")
    print(f"{'cond':>10} {'top1_model':>11} {'top5_model':>11} "
          f"{'top1_corp':>10} {'logprob':>9} {'cos_h62':>8}")
    res = {}
    for c in conds:
        a = {k: v / n_pos for k, v in acc[c].items()}
        res[c] = a
        print(f"{c:>10} {a['t1m']:>11.4f} {a['t5m']:>11.4f} {a['t1c']:>10.4f} "
              f"{a['lp']:>9.4f} {a['cos62']:>8.4f}")
    json.dump({"n_pos": n_pos, "res": res,
               "b_norms": {c: float(b[c].norm()) for c in mats}},
              open(f"{AUD}/readout3_audit.json", "w"), indent=2)
    print(f"wrote {AUD}/readout3_audit.json")


if __name__ == "__main__":
    main()
