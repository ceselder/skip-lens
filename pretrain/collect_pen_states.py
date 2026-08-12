"""Collect the REAL penultimate states at K future positions (skip-lens arm D).

Training inputs = h62[p], h62[p+1], ..., h62[p+K-1] — the model's ACTUAL
penultimate activations at the K positions whose tokens the decoder must emit.
No Jacobian, no autodiff: one plain forward per batch over prefix+rollout.

Why this and not transports: at test time the K slots are
Jbar^(d) @ h42[p], which (the J-bar audit measured cos(Jbar^0 h42, h62) = +0.50)
is an ESTIMATE OF h62[p+d] rather than a tangent. So the honest training target
is the thing being estimated — the real penultimate state — which is also the
skip-lens setup: train on the deepest honest representation, then feed a
shallower layer's transported estimate and see what is already present there.

Reuses pass-1 shards (prefix + ON-POLICY rollout + h42), so no re-rollout.

  python pretrain/collect_pen_states.py \
      --in-shards '/workspace/data/spans_raw/shard_*.parquet' \
      --out-dir /workspace/data/spans_pen8 --worker 0 --n-workers 1
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

N_OFFSETS = 16
SRC_LAYER = 42
TGT_LAYER = 62

OUT_SCHEMA_FIELDS = [
    ("doc_id", pa.string()),
    ("ctx_text", pa.string()),
    ("rollout_text", pa.string()),
    ("rollout_token_ids", pa.list_(pa.int32())),
    ("activation_vector", pa.list_(pa.float32())),   # h42 at p (test-time source)
    # [16, d] fp16 REAL h62 at p..p+15: np.frombuffer(b, np.float16).reshape(16, -1)
    ("transported_vectors", pa.binary()),
    ("transport_norms", pa.list_(pa.float32())),
    ("h42_recompute_cosine", pa.float32()),
]


def get_layers(model):
    return (model.model if hasattr(model, "model") else model).layers


def batch_rows(rows, batch_size):
    rows = sorted(rows, key=lambda r: len(r["teacher_input_ids"]))
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


@torch.no_grad()
def prepare_batch(batch, pad_id, device):
    seqs, p_list = [], []
    for r in batch:
        roll = r["rollout_token_ids"][0] if isinstance(
            r["rollout_token_ids"][0], list) else r["rollout_token_ids"]
        seqs.append(list(r["teacher_input_ids"]) + list(roll))
        p_list.append(len(r["teacher_input_ids"]) - 1)
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):          # RIGHT pad: harvest positions precede pads
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device), torch.tensor(p_list, device=device)


@torch.no_grad()
def future_states(model, ids, mask, p_pos):
    """([B, 16, d] real h62 at p..p+15, [B, d] recomputed h42 at p)."""
    layers = get_layers(model)
    store = {}
    hs = [layers[L].register_forward_hook(
        (lambda L: (lambda m, i, o: store.__setitem__(
            L, (o[0] if isinstance(o, tuple) else o).detach())))(L))
        for L in (SRC_LAYER, TGT_LAYER)]
    try:
        model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        for h in hs:
            h.remove()
    rows = torch.arange(ids.shape[0], device=ids.device)
    offs = torch.arange(N_OFFSETS, device=ids.device)
    gather = p_pos[:, None] + offs[None, :]
    return (store[TGT_LAYER][rows[:, None], gather].float(),
            store[SRC_LAYER][rows, p_pos].float())


def write_rows(out_path, rows_out):
    cols = {n: [r[n] for r in rows_out] for n, _ in OUT_SCHEMA_FIELDS}
    table = pa.table({n: pa.array(cols[n], type=t) for n, t in OUT_SCHEMA_FIELDS})
    tmp = str(out_path) + ".tmp"
    pq.write_table(table, tmp, row_group_size=1000)
    os.replace(tmp, out_path)
    return table.num_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--in-shards", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit-rows", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").cuda().eval()
    torch.set_grad_enabled(False)

    shards = sorted(glob.glob(args.in_shards))
    assert shards, f"no shards match {args.in_shards}"
    os.makedirs(args.out_dir, exist_ok=True)

    for shard in shards[args.worker :: args.n_workers]:
        out_path = Path(args.out_dir) / (Path(shard).stem + "_jvp.parquet")
        if out_path.exists():
            print(f"[skip] {out_path}", flush=True)
            continue
        rows = pq.read_table(shard).to_pylist()
        if args.limit_rows:
            rows = rows[: args.limit_rows]
        rows = [r for r in rows if r["rollout_token_ids"]
                and len(r["rollout_token_ids"][0]) >= N_OFFSETS]
        print(f"[{Path(shard).name}] {len(rows)} rows", flush=True)

        out = []
        for bi, batch in enumerate(batch_rows(rows, args.batch_size)):
            ids, mask, p_pos = prepare_batch(batch, pad_id, "cuda")
            states, h42_re = future_states(model, ids, mask, p_pos)
            stored = torch.tensor(np.array([r["act_L42"] for r in batch],
                                           dtype=np.float32)).cuda()
            cos = torch.nn.functional.cosine_similarity(h42_re, stored, dim=-1)
            norms = states.norm(dim=-1)
            for i, r in enumerate(batch):
                out.append({
                    "doc_id": r["doc_id"], "ctx_text": r["ctx_text"],
                    "rollout_text": (r["rollouts"] or [""])[0] if r.get("rollouts") else "",
                    "rollout_token_ids": [int(x) for x in r["rollout_token_ids"][0]],
                    "activation_vector": [float(x) for x in r["act_L42"]],
                    "transported_vectors": states[i].cpu().numpy()
                        .astype(np.float16).tobytes(),
                    "transport_norms": norms[i].cpu().numpy().astype(np.float32).tolist(),
                    "h42_recompute_cosine": float(cos[i]),
                })
            if bi % 40 == 0:
                print(f"  batch {bi}: |h62 p|={norms[:,0].mean():.1f} "
                      f"|h62 p+7|={norms[:,7].mean():.1f} h42cos={cos.mean():.4f}",
                      flush=True)
        print(f"[done] {out_path}: {write_rows(out_path, out)} rows", flush=True)


if __name__ == "__main__":
    main()
