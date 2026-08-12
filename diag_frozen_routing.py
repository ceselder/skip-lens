"""Is the local↔averaged transport gap caused by CONTENT-DEPENDENT ROUTING?

The multi-slot lens fails because the corpus-averaged transport J̄⁽ᵈ⁾ retains
only ~0.10-0.16 directional alignment with the true per-example local transport
at d>=1 (0.41 at d=0). Decompose the local transport as

    local = routing-dependent part  +  routing-independent part

where "routing" = the content-dependent gates the derivative flows through:
softmax attention weights, GatedDeltaNet gates, RMSNorm denominators, SiLU
gates. Averaging over contexts should kill the routing-dependent part and keep
the routing-independent one. If so, transports computed with those gates
DETACHED ("frozen routing", the chord/LRP linearization) should align far
better with J̄⁽ᵈ⁾ than ordinary local transports do.

Why it matters: if frozen-routing local transports align well, we can shrink
the train/test mismatch entirely from the TRAINING side (collect pass 2 with
frozen routing) while leaving the test-time averaged-J̄ readout untouched.

Reports, per delta: cos(local, J̄h), cos(frozen local, J̄h), and
cos(local, frozen local), plus across-example self-similarity of each.

Run in the fla-free venv (forward-mode AD needs the pure-torch DeltaNet path):
  /workspace/venv_jvp/bin/python diag_frozen_routing.py
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/skip-lens")
from pretrain.collect_jvp_transport import (  # noqa: E402
    SRC_LAYER,
    disable_tf32,
    jvp_transports,
    prepare_batch,
)

BASE = "Qwen/Qwen3.6-27B"
JDIR = os.environ.get("JBAR_DIR", "/workspace/results/offset_jlens")
# RAW shard: pass-2 output lacks teacher_input_ids, and note the raw
# `activation_vector` column is act_L62 by convention — h42 must come from
# act_L42 explicitly (per the data audit).
SHARD = os.environ.get("SHARD", "/workspace/data/spans_raw/shard_3.parquet")
N_ROWS = int(os.environ.get("N_ROWS", "64"))
BATCH = int(os.environ.get("BATCH", "8"))
DELTAS = [0, 1, 2, 3, 7]
OUT = os.environ.get("OUT", "/workspace/results/multislot_eval/frozen_routing.json")

# rlens_fit provides the class-level detach patches for Qwen3_5 (chord rules:
# RMSNorm denominator, SiLU/MLP gate, softmax attention weights, DeltaNet gates)
sys.path.insert(0, "/workspace/skip-lens")
import rlens_fit  # noqa: E402

# already @contextlib.contextmanager-decorated in rlens_fit: call it directly
_patch_cm = rlens_fit.lrp_detach_patches

disable_tf32()
tok = AutoTokenizer.from_pretrained(BASE)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
for p in model.parameters():
    p.requires_grad_(False)

rows = pq.read_table(SHARD).to_pylist()[:N_ROWS * 2]
rows = [r for r in rows if r["rollout_token_ids"]
        and len(r["rollout_token_ids"][0]) >= 16][:N_ROWS]
print(f"{len(rows)} rows from {SHARD}", flush=True)

Jbar = {d: torch.from_numpy(np.load(f"{JDIR}/Jbar_L42_to_L62_off{d}.npy")).float().cuda()
        for d in DELTAS}

loc_all, frz_all, jb_all = [], [], []
for c0 in range(0, len(rows), BATCH):
    batch = rows[c0:c0 + BATCH]
    ids, mask, p_pos = prepare_batch(batch, pad_id, "cuda")
    h42 = torch.tensor(np.array([r["act_L42"] for r in batch],
                                dtype=np.float32)).cuda()
    loc, _, _ = jvp_transports(model, ids, mask, p_pos, h42)
    with _patch_cm():
        frz, _, _ = jvp_transports(model, ids, mask, p_pos, h42)
    loc_all.append(loc.cpu())
    frz_all.append(frz.cpu())
    jb_all.append(torch.stack([ (Jbar[d] @ h42.T).T.cpu() for d in DELTAS ], dim=1))
    print(f"  batch {c0 // BATCH}: |loc d0|={loc[:,0].norm(dim=-1).mean():.1f} "
          f"|frz d0|={frz[:,0].norm(dim=-1).mean():.1f}", flush=True)

LOC = torch.cat(loc_all)            # [N, 16, d]
FRZ = torch.cat(frz_all)
JB = torch.cat(jb_all)              # [N, len(DELTAS), d]


def cos(a, b):
    c = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
    return float(c.mean()), float(c.std())


def self_sim(X):
    U = torch.nn.functional.normalize(X.float(), dim=-1)
    C = U @ U.T
    n = X.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    return float(C[iu[0], iu[1]].mean())


res = {}
print("\n delta | cos(local,Jbar) | cos(FROZEN,Jbar) | cos(local,frozen) | "
      "selfsim local | selfsim frozen | selfsim Jbar", flush=True)
for i, d in enumerate(DELTAS):
    lj = cos(LOC[:, d], JB[:, i])
    fj = cos(FRZ[:, d], JB[:, i])
    lf = cos(LOC[:, d], FRZ[:, d])
    row = {"cos_local_jbar": lj, "cos_frozen_jbar": fj, "cos_local_frozen": lf,
           "selfsim_local": self_sim(LOC[:, d]),
           "selfsim_frozen": self_sim(FRZ[:, d]),
           "selfsim_jbar": self_sim(JB[:, i])}
    res[d] = row
    print(f"   {d:2d}  |     {lj[0]:+.3f}      |      {fj[0]:+.3f}       |"
          f"      {lf[0]:+.3f}       |     {row['selfsim_local']:+.3f}    |"
          f"     {row['selfsim_frozen']:+.3f}    |    {row['selfsim_jbar']:+.3f}",
          flush=True)

json.dump({str(k): v for k, v in res.items()}, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}", flush=True)
gain = res[3]["cos_frozen_jbar"][0] - res[3]["cos_local_jbar"][0]
print("VERDICT:", f"FROZEN ROUTING HELPS (+{gain:.3f} at d=3) — worth "
      "recollecting pass 2 with frozen routing" if gain > 0.15 else
      f"no material gain ({gain:+.3f} at d=3) — the gap is not (only) routing; "
      "averaged per-horizon transports lack per-example content intrinsically",
      flush=True)
