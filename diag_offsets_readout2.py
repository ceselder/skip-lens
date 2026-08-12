"""AUDIT follow-up (GPU, main venv): decompose the T1 result.

diag_offsets_readout.py found decode(Jbar^0 h42) reads out WORSE than the
plain L42 logit lens. Here we compare, at target offset 0 (next token at t):
    J0        shipped Jbar^(0)
    J0T       its transpose (sign/orientation control)
    offpooled shipped mean over the 16 offsets
    sum16     sum over the 16 offsets (approximates the paper's pooled
              estimator J = E[sum_{t'>=t} dh62[t']/dh42[t]] up to the >15 tail)
    h42lens   identity transport (plain logit lens)
    h62true   oracle
plus activation-space stats incl. the identity-transport baseline
cos(h42[t], h62[t]) that the first script omitted.

Run:  CUDA_VISIBLE_DEVICES=2 /workspace/venv/bin/python diag_offsets_readout2.py
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
N_PROMPTS = int(os.environ.get("N_PROMPTS", "24"))


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
        torch.from_numpy(np.load(f"{OUT}/Jbar_L{SRC}_to_L{TGT}_off{d}.npy"))
        for d in range(16)]).cuda()
    mats = {
        "J0": Jall[0],
        "J0T": Jall[0].T.contiguous(),
        "offpooled": Jall.mean(0),
        "sum16": Jall.sum(0),
    }
    del Jall
    prompts = json.load(open(f"{AUD}/heldout_prompts.json"))[:N_PROMPTS]

    conds = list(mats) + ["h42lens", "h62true"]
    acc = {c: {"t1m": 0.0, "t5m": 0.0, "t1c": 0.0, "t5c": 0.0, "lp": 0.0,
               "cos62": 0.0} for c in conds}
    n_pos = 0

    store = {}
    def cap(idx):
        def hook(m, i, o):
            store[idx] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook

    hooks = [tm.layers[SRC].register_forward_hook(cap(SRC)),
             tm.layers[TGT].register_forward_hook(cap(TGT))]
    try:
        for pi, prompt in enumerate(prompts):
            ids = model.encode(prompt, max_length=SEQ)
            with torch.no_grad():
                out = hf(input_ids=ids, use_cache=False)
            final_logits = out.logits[0].float()
            h42 = store[SRC][0].float()
            h62 = store[TGT][0].float()
            seq_len = ids.shape[1]
            ts = torch.arange(T_LO, seq_len - 2, STRIDE, device="cuda")
            ids0 = ids[0]
            nxt = ids0[ts + 1]
            model_next = final_logits[ts].argmax(-1)
            h42_t, h62_t = h42[ts], h62[ts]

            rows = {c: (h42_t @ m.T) for c, m in mats.items()}
            rows["h42lens"] = h42_t
            rows["h62true"] = h62_t
            for c, x in rows.items():
                lg = decode(x)
                lsm = lg - torch.logsumexp(lg, -1, keepdim=True)
                am = lg.argmax(-1)
                t5 = lg.topk(5, -1).indices
                acc[c]["t1m"] += (am == model_next).double().sum().item()
                acc[c]["t5m"] += (t5 == model_next[:, None]).any(-1).double().sum().item()
                acc[c]["t1c"] += (am == nxt).double().sum().item()
                acc[c]["t5c"] += (t5 == nxt[:, None]).any(-1).double().sum().item()
                acc[c]["lp"] += lsm.gather(1, nxt[:, None]).double().sum().item()
                acc[c]["cos62"] += F.cosine_similarity(x, h62_t, -1).double().sum().item()
                del lg, lsm
            n_pos += len(ts)
            print(f"prompt {pi+1}/{len(prompts)} total={n_pos}", flush=True)
    finally:
        for h in hooks:
            h.remove()

    print(f"\n#### readout2: next-token @ t (n={n_pos}, {len(prompts)} prompts)")
    print(f"{'cond':>10} {'top1_model':>11} {'top5_model':>11} {'top1_corp':>10} "
          f"{'top5_corp':>10} {'logprob':>9} {'cos_h62':>8}")
    res = {}
    for c in conds:
        a = {k: v / n_pos for k, v in acc[c].items()}
        res[c] = a
        print(f"{c:>10} {a['t1m']:>11.4f} {a['t5m']:>11.4f} {a['t1c']:>10.4f} "
              f"{a['t5c']:>10.4f} {a['lp']:>9.4f} {a['cos62']:>8.4f}")
    json.dump({"n_pos": n_pos, "res": res},
              open(f"{AUD}/readout2_audit.json", "w"), indent=2)
    print(f"wrote {AUD}/readout2_audit.json")


if __name__ == "__main__":
    main()
