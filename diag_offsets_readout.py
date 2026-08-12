"""AUDIT (GPU, main venv): token-space + activation-space tests of Jbar^(delta).

T1  delta=0 J-lens readout quality: decode(Jbar^0 @ h42[t]) vs plain L42
    logit lens vs true-L62 logit lens, scored against (a) the model's own
    argmax next token and (b) the corpus next token.
T2  Offset indexing: rows = matrix delta (0..7), cols = target offset
    delta' (0..7). Cell = mean log-prob / top-1 acc of the CORPUS token at
    t+delta'+1 under decode(Jbar^(delta) @ h42[t]); plus an activation-space
    matrix cos(Jbar^(delta) h42[t], h62[t+delta']) raw and centered.
    Correct indexing <=> every row peaks on its own diagonal.
T3  Sign/transpose: cos(J0 h42, h62[t]) vs cos(J0.T h42, h62[t]) vs
    cos(-J0 h42, h62[t]); norm ratios.

Held-out prompts (written by diag_offsets_static.py). Decode uses the final
RMSNorm with the empirically-probed (1+w) gain from interface/common.py.

Run:  CUDA_VISIBLE_DEVICES=1 /workspace/venv/bin/python diag_offsets_readout.py
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/workspace/skip-lens")
from transformers import AutoModelForCausalLM, AutoTokenizer

from interface.common import norm_gain, resolve_text_model  # read-only import
from jlens.hf import from_hf

OUT = "/workspace/results/offset_jlens"
AUD = "/workspace/results/offset_jlens_audit"
BASE = "Qwen/Qwen3.6-27B"
SRC, TGT = 42, 62
SEQ = 512
MAXD = 8          # deltas 0..7 in the T2 matrix
T_LO, STRIDE = 31, 3
N_PROMPTS = int(os.environ.get("N_PROMPTS", "24"))


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    hf = (AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        .cuda().eval())
    model = from_hf(hf, tok)  # matches fit-time encode() incl. BOS handling
    tm = resolve_text_model(hf)
    gain = norm_gain(hf).cuda()  # fp32 [d], validated (1+w) vs w numerically
    eps = float(getattr(hf.config.get_text_config(), "rms_norm_eps", 1e-6))
    softcap = getattr(hf.config.get_text_config(), "final_logit_softcapping", None)
    W_U = hf.lm_head.weight.detach().float().cuda()  # [V, d]
    print(f"rms_norm_eps={eps} softcap={softcap} gain[:3]={gain[:3].tolist()}")

    def decode(x: torch.Tensor) -> torch.Tensor:
        """Final-norm logit lens: x [n, d] fp32 -> logits [n, V] fp32."""
        u = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        logits = (u * gain) @ W_U.T
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    Js = torch.stack([
        torch.from_numpy(np.load(f"{OUT}/Jbar_L{SRC}_to_L{TGT}_off{d}.npy"))
        for d in range(MAXD)
    ]).cuda()  # [MAXD, d, d] fp32
    J0 = Js[0]

    prompts = json.load(open(f"{AUD}/heldout_prompts.json"))[:N_PROMPTS]

    conds = [f"J{d}" for d in range(MAXD)] + ["h42lens", "h62true"]
    n_conds = len(conds)
    # accumulators
    lp_sum = torch.zeros(n_conds, MAXD, dtype=torch.float64)   # mean logprob
    top1_sum = torch.zeros(n_conds, MAXD, dtype=torch.float64)
    top1_model = torch.zeros(n_conds, dtype=torch.float64)     # vs model argmax @ t
    top5_model = torch.zeros(n_conds, dtype=torch.float64)
    top5_corpus = torch.zeros(n_conds, dtype=torch.float64)    # corpus tok t+1
    n_pos = 0
    # activation-space moments for cos matrices (fp64)
    d_model = Js.shape[-1]
    m_pt = torch.zeros(MAXD, MAXD, dtype=torch.float64)  # E[pred . tgt]
    m_pp = torch.zeros(MAXD, dtype=torch.float64)        # E[|pred|^2]
    m_tt = torch.zeros(MAXD, dtype=torch.float64)        # E[|tgt|^2]
    s_p = torch.zeros(MAXD, d_model, dtype=torch.float64)
    s_t = torch.zeros(MAXD, d_model, dtype=torch.float64)
    cos_raw = torch.zeros(MAXD, MAXD, dtype=torch.float64)     # mean per-sample cos
    # T3
    t3 = {k: [] for k in ("cos_J0h", "cos_J0Th", "cos_negJ0h",
                          "norm_J0h", "norm_h62", "norm_h42")}

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
            seq_len = ids.shape[1]
            with torch.no_grad():
                out = hf(input_ids=ids, use_cache=False)
            final_logits = out.logits[0].float()          # [T, V]
            h42 = store[SRC][0].float()                   # [T, d]
            h62 = store[TGT][0].float()
            ts = torch.arange(T_LO, seq_len - 1 - MAXD, STRIDE, device="cuda")
            n = len(ts)
            ids0 = ids[0]
            tgt_tok = torch.stack([ids0[ts + 1 + dp] for dp in range(MAXD)], 1)  # [n, MAXD]
            model_next = final_logits[ts].argmax(-1)                             # [n]

            h42_t = h42[ts]                               # [n, d]
            preds = torch.einsum("kij,nj->kni", Js, h42_t)  # [MAXD, n, d]
            rows = torch.cat([preds, h42_t[None], h62[ts][None]], 0)  # [n_conds, n, d]

            for ci in range(n_conds):
                lg = decode(rows[ci])                     # [n, V]
                lsm = lg - torch.logsumexp(lg, -1, keepdim=True)
                lp_sum[ci] += lsm.gather(1, tgt_tok).sum(0).double().cpu()
                am = lg.argmax(-1)
                top1_sum[ci] += (am[:, None] == tgt_tok).double().sum(0).cpu()
                t5 = lg.topk(5, -1).indices               # [n, 5]
                top1_model[ci] += (am == model_next).double().sum().cpu()
                top5_model[ci] += (t5 == model_next[:, None]).any(-1).double().sum().cpu()
                top5_corpus[ci] += (t5 == tgt_tok[:, :1]).any(-1).double().sum().cpu()
                del lg, lsm

            # activation-space matrices
            tgts = torch.stack([h62[ts + dp] for dp in range(MAXD)], 0)  # [MAXD, n, d]
            for d in range(MAXD):
                p = preds[d]
                for dp in range(MAXD):
                    t_ = tgts[dp]
                    cos_raw[d, dp] += F.cosine_similarity(p, t_, -1).double().sum().cpu()
                    m_pt[d, dp] += (p * t_).sum().double().cpu()
                m_pp[d] += p.pow(2).sum().double().cpu()
                m_tt[d] += tgts[d].pow(2).sum().double().cpu()
                s_p[d] += p.sum(0).double().cpu()
                s_t[d] += tgts[d].sum(0).double().cpu()

            # T3
            J0h = preds[0]
            J0Th = h42_t @ J0                              # = (J0.T @ h42)^T rows
            h62_t = h62[ts]
            t3["cos_J0h"].append(F.cosine_similarity(J0h, h62_t, -1).mean().item())
            t3["cos_J0Th"].append(F.cosine_similarity(J0Th, h62_t, -1).mean().item())
            t3["cos_negJ0h"].append(F.cosine_similarity(-J0h, h62_t, -1).mean().item())
            t3["norm_J0h"].append(J0h.norm(dim=-1).mean().item())
            t3["norm_h62"].append(h62_t.norm(dim=-1).mean().item())
            t3["norm_h42"].append(h42_t.norm(dim=-1).mean().item())

            n_pos += n
            print(f"prompt {pi+1}/{len(prompts)} seq={seq_len} n_pos={n} total={n_pos}",
                  flush=True)
    finally:
        for h in hooks:
            h.remove()

    lp = (lp_sum / n_pos).numpy()
    t1 = (top1_sum / n_pos).numpy()
    t1m = (top1_model / n_pos).numpy()
    t5m = (top5_model / n_pos).numpy()
    t5c = (top5_corpus / n_pos).numpy()
    craw = (cos_raw / n_pos).numpy()
    # centered alignment: corr of pred/tgt about their sample means
    mp, mt = s_p / n_pos, s_t / n_pos
    ccen = np.zeros((MAXD, MAXD))
    for d in range(MAXD):
        for dp in range(MAXD):
            num = m_pt[d, dp] / n_pos - (mp[d] * mt[dp]).sum()
            den = torch.sqrt((m_pp[d] / n_pos - mp[d].pow(2).sum())
                             * (m_tt[dp] / n_pos - mt[dp].pow(2).sum()))
            ccen[d, dp] = (num / den).item()

    def show(mat, name, fmt="{:8.4f}"):
        print(f"\n== {name} (rows = condition, cols = target offset delta') ==")
        print("        " + " ".join(f"{dp:>8d}" for dp in range(MAXD)) + "   rowargmax")
        for ci, cn in enumerate(conds[: mat.shape[0]]):
            cells = " ".join(fmt.format(v) for v in mat[ci])
            print(f"{cn:>7} {cells}   {int(np.argmax(mat[ci]))}")

    print(f"\n#### T1: delta=0 readout at t (n={n_pos} positions, "
          f"{len(prompts)} held-out prompts)")
    print(f"{'cond':>8} {'top1_vs_model':>14} {'top5_vs_model':>14} "
          f"{'top1_vs_corpus':>15} {'top5_vs_corpus':>15} {'logprob_corpus':>15}")
    for ci, cn in enumerate(conds):
        print(f"{cn:>8} {t1m[ci]:>14.4f} {t5m[ci]:>14.4f} "
              f"{t1[ci,0]:>15.4f} {t5c[ci]:>15.4f} {lp[ci,0]:>15.4f}")

    show(lp, "T2a mean log-prob of corpus token @ t+delta'+1")
    show(t1, "T2b top-1 acc of corpus token @ t+delta'+1")
    show(craw[:MAXD], "T2c raw cos(J^d h42[t], h62[t+delta'])")
    show(ccen, "T2d centered cos (corr) matrix")

    print("\n#### T3: sign/transpose sanity (per-prompt means)")
    for k, v in t3.items():
        print(f"{k:>10}: mean={np.mean(v):.4f} sd={np.std(v):.4f}")

    json.dump({
        "n_pos": n_pos, "n_prompts": len(prompts), "conds": conds,
        "t1": {"top1_vs_model": t1m.tolist(), "top5_vs_model": t5m.tolist(),
               "top5_vs_corpus": t5c.tolist()},
        "t2_logprob": lp.tolist(), "t2_top1": t1.tolist(),
        "t2_cos_raw": craw.tolist(), "t2_cos_centered": ccen.tolist(),
        "t3": {k: [float(np.mean(v)), float(np.std(v))] for k, v in t3.items()},
    }, open(f"{AUD}/readout_audit.json", "w"), indent=2)
    print(f"\nwrote {AUD}/readout_audit.json")


if __name__ == "__main__":
    main()
