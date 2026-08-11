"""Localize the qwen3_5 fwd/rev AD discrepancy seen in the JVP selftest.

On tiny Llama: jvp == dvjp == single-vjp probe (1e-6). On Qwen3.6-27B:
jvp == dvjp but both differ from the probe by 25-124%. A pure-torch backward
is linear in its cotangent, so the identity SHOULD hold — this script
separates the three remaining explanations:

  A. same-graph: on ONE forward's graph, compare <u, Jv> via double-backward
     against <J^T u, v> via plain backward. Disagreement = an op whose
     double-backward is silently wrong / cotangent-nonlinear.
  B. call-to-call: run the plain probe on two separate forwards. Drift =
     module state mutated between calls (lazy caches, buffers).
  C. batch-shape: dvjp on batch-of-1 vs row 0 of dvjp on batch-of-4.
     Difference = padding/mask/kernel-shape dependence.

Run on the box (bf16, one GPU, fits alongside collection):
  python diag_dvjp_probe.py /workspace/data/spans_smoke/shard_smoke.parquet
"""
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pretrain.collect_jvp_transport import (
    N_OFFSETS,
    SRC_LAYER,
    TGT_LAYER,
    dvjp_transports,
    get_layers,
    prepare_batch,
)

BASE = "Qwen/Qwen3.6-27B"
shard = sys.argv[1]

tok = AutoTokenizer.from_pretrained(BASE)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)
layers = get_layers(model)

rows = pq.read_table(shard).to_pylist()[:4]
rows = [r for r in rows if r["rollout_token_ids"]
        and len(r["rollout_token_ids"][0]) >= N_OFFSETS]
ids4, mask4, p4 = prepare_batch(rows, pad_id, "cuda")
stored4 = torch.tensor(
    np.array([r[f"act_L{SRC_LAYER}"] for r in rows], dtype=np.float32)).cuda()
# NOTE prepare_batch sorts internally; for this diagnostic only pairing
# consistency matters, and every check below builds its own pairing.

ids1, mask1, p1 = ids4[:1], mask4[:1], p4[:1]
v = stored4[:1].to(torch.bfloat16)
DELTAS = (0, 3, 8, 15)


def forward_rooted(ids, mask):
    store = {}

    def cap42(m, i, o):
        t = o[0] if isinstance(o, tuple) else o
        t.requires_grad_(True)
        store["h42"] = t

    def cap62(m, i, o):
        store["h62"] = o[0] if isinstance(o, tuple) else o

    h1 = layers[SRC_LAYER].register_forward_hook(cap42)
    h2 = layers[TGT_LAYER].register_forward_hook(cap62)
    try:
        with torch.enable_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        h1.remove()
        h2.remove()
    return store["h42"], store["h62"]


print("=== A. same-graph double-backward vs plain backward ===", flush=True)
h42, h62 = forward_rooted(ids1, mask1)
p = int(p1[0])
with torch.enable_grad():
    w = torch.zeros_like(h62, requires_grad=True)
    s = (h62 * w).sum()
    (g42_w,) = torch.autograd.grad(s, h42, create_graph=True)
    s2 = (g42_w[0, p] * v[0]).sum()
    (jv,) = torch.autograd.grad(s2, w, retain_graph=True)
    for d in DELTAS:
        gen = torch.Generator().manual_seed(d)
        u = torch.randn(h62.shape[-1], generator=gen).cuda()
        lhs = float((jv[0, p + d].float() * u).sum())      # <u, Jv> via dvjp
        out = (h62[0, p + d].float() * u).sum()
        (g_u,) = torch.autograd.grad(out, h42, retain_graph=True)
        rhs = float((g_u[0, p].float() * v[0].float()).sum())  # <J^T u, v>
        rel = abs(lhs - rhs) / (abs(rhs) + 1e-9)
        print(f"  delta={d}: lhs={lhs:+.4e} rhs={rhs:+.4e} rel={rel:.2e}",
              flush=True)

print("=== B. call-to-call stability of the plain probe ===", flush=True)
vals = []
for rep in range(2):
    h42b, h62b = forward_rooted(ids1, mask1)
    with torch.enable_grad():
        per = []
        for d in DELTAS:
            gen = torch.Generator().manual_seed(d)
            u = torch.randn(h62b.shape[-1], generator=gen).cuda()
            out = (h62b[0, p + d].float() * u).sum()
            (g_u,) = torch.autograd.grad(out, h42b, retain_graph=True)
            per.append(float((g_u[0, p].float() * v[0].float()).sum()))
        vals.append(per)
for i, d in enumerate(DELTAS):
    a, b = vals[0][i], vals[1][i]
    print(f"  delta={d}: run1={a:+.4e} run2={b:+.4e} "
          f"rel={abs(a - b) / (abs(b) + 1e-9):.2e}", flush=True)

print("=== C. batch-of-1 vs batch-of-4 dvjp (row 0) ===", flush=True)
t1, _ = dvjp_transports(model, ids1, mask1, p1, stored4[:1])
t4, _ = dvjp_transports(model, ids4, mask4, p4, stored4)
cos = torch.nn.functional.cosine_similarity(t1[0], t4[0], dim=-1)
print("  per-delta cos:", " ".join(f"{float(c):.4f}" for c in cos), flush=True)
print("DONE", flush=True)
