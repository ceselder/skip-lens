"""AUDIT (GPU, venv_jvp): ground-truth check of the comb+Rademacher estimator.

For a few held-out prompts and random unit vectors v, computes EXACT local
Jacobian-vector products  J_local^(delta)[t] @ v = d h62[t+delta]/d h42[t] @ v
via double-VJP (reverse-over-reverse — the same reverse-mode semantics as the
comb estimator; same construction as pretrain/collect_jvp_transport.py, which
must run without flash-linear-attention and with eager attention).

The empirical average of these local JVPs over (prompt, t) pairs must converge
to Jbar^(delta) @ v. We report, per delta:
  - cos(mean_n JVP, Jbar^(delta) v) for growing n  (convergence)
  - norm ratio at max n                            (scale)
  - full 16x16 cos matrix mean_JVP[delta] vs Jbar^(delta') v  (indexing,
    completely independent of any token-space convention)

Run:  CUDA_VISIBLE_DEVICES=2 /workspace/venv_jvp/bin/python diag_offsets_jvp.py
"""
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = "/workspace/results/offset_jlens"
AUD = "/workspace/results/offset_jlens_audit"
BASE = "Qwen/Qwen3.6-27B"
SRC, TGT = 42, 62
N_OFFSETS = 16
SEQ = 512
N_PROMPTS = int(os.environ.get("N_PROMPTS", "4"))
POS_PER_PROMPT = int(os.environ.get("POS_PER_PROMPT", "64"))
N_VECS = 2
BATCH = int(os.environ.get("BATCH", "8"))
VSCALE = 100.0  # keep bf16 activations in a comfortable range; divided out


def get_layers(model):
    return (model.model if hasattr(model, "model") else model).layers


def dvjp_transports(model, ids, p_pos, tangent_vecs):
    """[B, N_OFFSETS, d] exact local transports, reverse-over-reverse.

    Same math as pretrain/collect_jvp_transport.py::dvjp_transports:
    J v = d/dw < d<h62, w>/dh42[p], v > — linear in w, so the outer grad is
    the exact directional derivative with pure reverse-mode rules.
    """
    layers = get_layers(model)
    dtype = next(model.parameters()).dtype
    tangent = tangent_vecs.to(ids.device, dtype)
    rows = torch.arange(ids.shape[0], device=ids.device)
    store = {}

    def cap42(module, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        t.requires_grad_(True)  # params frozen -> this roots the graph
        store["h42"] = t

    def cap62(module, inputs, output):
        store["h62"] = output[0] if isinstance(output, tuple) else output

    h1 = layers[SRC].register_forward_hook(cap42)
    h2 = layers[TGT].register_forward_hook(cap62)
    try:
        with torch.enable_grad():
            model(input_ids=ids, use_cache=False)
            h42, h62 = store["h42"], store["h62"]
            w = torch.zeros_like(h62, requires_grad=True)
            s = (h62 * w).sum()
            (g42,) = torch.autograd.grad(s, h42, create_graph=True)
            s2 = (g42[rows, p_pos] * tangent).sum()
            (jv,) = torch.autograd.grad(s2, w)
    finally:
        h1.remove()
        h2.remove()
    offs = torch.arange(N_OFFSETS, device=ids.device)
    gather = p_pos[:, None] + offs[None, :]
    return jv.detach()[rows[:, None], gather].float()


def main() -> None:
    try:
        import fla  # noqa: F401
        raise SystemExit("fla installed — wrong venv, use /workspace/venv_jvp")
    except ImportError:
        pass

    tok = AutoTokenizer.from_pretrained(BASE)
    model = (AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, attn_implementation="eager")
        .cuda().eval())
    for p in model.parameters():
        p.requires_grad_(False)

    prompts = json.load(open(f"{AUD}/heldout_prompts.json"))[24:24 + N_PROMPTS]
    d_model = model.config.hidden_size

    gen = torch.Generator().manual_seed(0)
    vs = torch.randn(N_VECS, d_model, generator=gen)
    vs = vs / vs.norm(dim=-1, keepdim=True)

    # reference: Jbar^(delta) @ v, fp32
    Jbar = np.stack([np.load(f"{OUT}/Jbar_L{SRC}_to_L{TGT}_off{d}.npy")
                     for d in range(N_OFFSETS)])  # [16, d, d]
    ref = np.einsum("kij,vj->vki", Jbar, vs.numpy())  # [V, 16, d]

    rng = np.random.default_rng(0)
    samples = {v: [] for v in range(N_VECS)}  # list of [16, d] arrays
    for pi, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=SEQ).input_ids.cuda()
        seq_len = ids.shape[1]
        pos = rng.choice(np.arange(31, seq_len - N_OFFSETS - 1),
                         size=POS_PER_PROMPT, replace=False)
        for vi in range(N_VECS):
            for i in range(0, POS_PER_PROMPT, BATCH):
                chunk = pos[i:i + BATCH]
                b = len(chunk)
                bids = ids.expand(b, -1)
                p_pos = torch.tensor(chunk, device="cuda")
                tang = (vs[vi] * VSCALE).unsqueeze(0).expand(b, -1).cuda()
                tr = dvjp_transports(model, bids, p_pos, tang) / VSCALE
                samples[vi].extend(t.cpu().numpy() for t in tr)
            print(f"prompt {pi+1}/{N_PROMPTS} v{vi}: {len(samples[vi])} samples",
                  flush=True)

    result = {"n_prompts": N_PROMPTS, "pos_per_prompt": POS_PER_PROMPT}
    all_cos_mat = np.zeros((N_VECS, N_OFFSETS, N_OFFSETS))
    for vi in range(N_VECS):
        arr = np.stack(samples[vi])  # [n, 16, d]
        order = rng.permutation(len(arr))
        arr = arr[order]
        np.save(f"{AUD}/jvp_samples_v{vi}.npy", arr.astype(np.float32))
        print(f"\n==== v{vi}: {len(arr)} local JVP samples ====")
        curves = {}
        for n in (8, 32, 128, len(arr)):
            m = arr[:n].mean(0)  # [16, d]
            cs = [float(np.dot(m[d], ref[vi, d])
                        / (np.linalg.norm(m[d]) * np.linalg.norm(ref[vi, d])))
                  for d in range(N_OFFSETS)]
            curves[n] = cs
            print(f"cos(mean_{n:>3} JVP, Jbar v) per delta: "
                  + " ".join(f"{c:.3f}" for c in cs))
        m = arr.mean(0)
        ratio = [float(np.linalg.norm(m[d]) / np.linalg.norm(ref[vi, d]))
                 for d in range(N_OFFSETS)]
        print("norm ratio |mean JVP| / |Jbar v| per delta: "
              + " ".join(f"{r:.3f}" for r in ratio))
        # indexing matrix: empirical mean at delta vs Jbar^(delta') v
        for d in range(N_OFFSETS):
            for dp in range(N_OFFSETS):
                all_cos_mat[vi, d, dp] = float(
                    np.dot(m[d], ref[vi, dp])
                    / (np.linalg.norm(m[d]) * np.linalg.norm(ref[vi, dp])))
        print("indexing: argmax over delta' of cos(meanJVP[delta], Jbar^(delta') v):",
              [int(np.argmax(all_cos_mat[vi, d])) for d in range(N_OFFSETS)])
        result[f"v{vi}"] = {"curves": {str(k): v for k, v in curves.items()},
                            "norm_ratio": ratio,
                            "cos_matrix": all_cos_mat[vi].tolist()}

    json.dump(result, open(f"{AUD}/jvp_audit.json", "w"), indent=2)
    print(f"\nwrote {AUD}/jvp_audit.json")


if __name__ == "__main__":
    main()
