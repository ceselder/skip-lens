"""AUDIT (GPU, venv_jvp): EXACT single-tooth comb vs autograd ground truth.

With comb_spacing=480 and a phase that leaves exactly ONE tooth at position
t*, jlens.offset_fitting.offset_jacobians_for_prompt returns, with zero
contamination and zero averaging,
    J_comb[delta][i, :] = sign^2 * d h62[t*, i] / d h42[t* - delta, :]
i.e. the exact local Jacobian rows. Independently, double-VJP gives
    gt[delta] = d h62[t*] / d h42[t* - delta] @ v
for the same (prompt, positions). If tooth placement, the Rademacher sign
handling, or the offset extraction indexing were wrong in ANY way,
J_comb[delta] @ v != gt[delta]. Both computed in THIS venv (eager attention,
pure-torch DeltaNet), so the only slack is bf16 accumulation noise.

Also cross-indexes: cos(J_comb[delta] v, gt[delta']) must peak at delta'=delta.

Run:  CUDA_VISIBLE_DEVICES=2 /workspace/venv_jvp/bin/python diag_offsets_exact.py
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from jlens.hf import from_hf
from jlens.offset_fitting import offset_jacobians_for_prompt

AUD = "/workspace/results/offset_jlens_audit"
BASE = "Qwen/Qwen3.6-27B"
SRC, TGT = 42, 62
N_OFFSETS = 16
SEQ = 512
SPACING = 480
PHASE = 200            # single tooth at t* = 16 + 15 + 200 = 231
SIGN_SEED = 7
DIM_BATCH = int(os.environ.get("DIM_BATCH", "16"))
BATCH = int(os.environ.get("BATCH", "4"))
N_VECS = 2
VSCALE = 100.0


def get_layers(model):
    return (model.model if hasattr(model, "model") else model).layers


def dvjp(model, ids, p_pos, tangent_vecs):
    layers = get_layers(model)
    dtype = next(model.parameters()).dtype
    tangent = tangent_vecs.to(ids.device, dtype)
    rows = torch.arange(ids.shape[0], device=ids.device)
    store = {}

    def cap42(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        t.requires_grad_(True)
        store["h42"] = t

    def cap62(m, i, o):
        store["h62"] = o[0] if isinstance(o, tuple) else o

    h1 = layers[SRC].register_forward_hook(cap42)
    h2 = layers[TGT].register_forward_hook(cap62)
    try:
        with torch.enable_grad():
            model(input_ids=ids, use_cache=False)
            h42, h62 = store["h42"], store["h62"]
            w = torch.zeros_like(h62, requires_grad=True)
            (g42,) = torch.autograd.grad((h62 * w).sum(), h42, create_graph=True)
            (jv,) = torch.autograd.grad((g42[rows, p_pos] * tangent).sum(), w)
    finally:
        h1.remove()
        h2.remove()
    return jv.detach()  # [B, T, d]


def main() -> None:
    try:
        import fla  # noqa: F401
        raise SystemExit("fla installed — wrong venv, use /workspace/venv_jvp")
    except ImportError:
        pass

    tok = AutoTokenizer.from_pretrained(BASE)
    hf = (AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="eager")
        .cuda().eval())
    for p in hf.parameters():
        p.requires_grad_(False)
    model = from_hf(hf, tok)
    prompt = json.load(open(f"{AUD}/heldout_prompts.json"))[28]
    tstar = 16 + N_OFFSETS - 1 + PHASE

    # --- comb estimator, single tooth (exact rows, no averaging) -----------
    cache = f"{AUD}/exact_singletooth_J.pt"
    if os.path.exists(cache):
        J = torch.load(cache, map_location="cpu")
        print(f"loaded cached single-tooth J from {cache}", flush=True)
    else:
        jac, seq_len, n_teeth = offset_jacobians_for_prompt(
            model, prompt, [SRC], target_layer=TGT,
            n_offsets=N_OFFSETS, comb_spacing=SPACING, dim_batch=DIM_BATCH,
            max_seq_len=SEQ, phase=PHASE, sign_seed=SIGN_SEED)
        J = jac[SRC]  # [16, d, d] fp32
        print(f"seq_len={seq_len} n_teeth={n_teeth} (must be 1) t*={tstar}",
              flush=True)
        assert n_teeth == 1
        torch.save(J, cache)

    d_model = J.shape[-1]
    gen = torch.Generator().manual_seed(0)
    vs = torch.randn(N_VECS, d_model, generator=gen)
    vs = vs / vs.norm(dim=-1, keepdim=True)

    # --- ground truth: dvjp at p = t*-delta, read target position t* -------
    ids = model.encode(prompt, max_length=SEQ)
    gt = torch.zeros(N_VECS, N_OFFSETS, d_model)
    for vi in range(N_VECS):
        for i in range(0, N_OFFSETS, BATCH):
            deltas = list(range(i, min(i + BATCH, N_OFFSETS)))
            b = len(deltas)
            p_pos = torch.tensor([tstar - d for d in deltas], device="cuda")
            tang = (vs[vi] * VSCALE).unsqueeze(0).expand(b, -1).cuda()
            jv = dvjp(hf, ids.expand(b, -1), p_pos, tang) / VSCALE
            gt[vi, deltas] = jv[torch.arange(b), tstar].float().cpu()
            print(f"v{vi} deltas {deltas} done", flush=True)

    # --- compare ------------------------------------------------------------
    res = []
    print(f"\n{'d':>2} {'v':>2} {'cos(Jcomb v, gt)':>17} {'rel_err':>9} "
          f"{'|Jcomb v|':>10} {'|gt|':>8} {'argmax_dp':>9}")
    for vi in range(N_VECS):
        pred = torch.einsum("kij,j->ki", J.double(), vs[vi].double())  # [16, d]
        for d in range(N_OFFSETS):
            g = gt[vi, d].double()
            c = float(torch.dot(pred[d], g) / (pred[d].norm() * g.norm()))
            rel = float((pred[d] - g).norm() / g.norm())
            cross = [float(torch.dot(pred[d], gt[vi, dp].double())
                           / (pred[d].norm() * gt[vi, dp].double().norm()))
                     for dp in range(N_OFFSETS)]
            am = int(np.argmax(cross))
            res.append({"v": vi, "delta": d, "cos": c, "rel_err": rel,
                        "norm_pred": float(pred[d].norm()),
                        "norm_gt": float(g.norm()), "argmax_dp": am})
            print(f"{d:>2} {vi:>2} {c:>17.5f} {rel:>9.4f} "
                  f"{pred[d].norm():>10.4f} {g.norm():>8.4f} {am:>9d}")

    json.dump({"tstar": tstar, "rows": res},
              open(f"{AUD}/exact_audit.json", "w"), indent=2)
    print(f"wrote {AUD}/exact_audit.json")


if __name__ == "__main__":
    main()
