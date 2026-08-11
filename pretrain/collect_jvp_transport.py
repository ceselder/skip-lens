"""Pass 2 of local-transport span collection: per-offset JVP transports.

For each pass-1 row (prefix ids, ON-POLICY 16-token rollout, h at L42 pos p),
computes the LOCAL Jacobian transports J_local^(delta) @ h for delta=0..15,
where J_local^(delta) = d h62[p+delta] / d h42[p] evaluated on the actual
prefix+rollout sequence. These are the arm-A training inputs; at test time
they are replaced by the corpus-AVERAGED Jbar^(delta) @ h (the deliberate
train/test mismatch that projects onto broadly-verbalizable content).

Implementation: one torch.func.jvp per BATCH. eps (scalar 0-d tensor) is the
only differentiated input; a forward hook on layers[42] adds
eps * h42_stored[row] at each row's position p, and the function returns the
hooked layers[62] output. The forward-mode tangent of that output at
positions p..p+15 is exactly [J_local^(delta) h]_delta — rows don't interact,
so one shared eps serves the whole batch. Forward-mode needs no retained
graph: cost ~2x a plain forward.

Kernel requirements: run WITHOUT flash-linear-attention/causal-conv1d
installed (qwen3_5 falls back to pure-torch DeltaNet) and with
attn_implementation="eager", so every op supports forward AD.

Self-test (--selftest): compares bf16 JVP against central finite differences
on an fp32 model copy for N rows; run once per box before fleet collection.

Leakage probe (--probe-frac): a fraction of rows is duplicated with tangents
taken from a DIFFERENT row (derangement within batch) and written to a
sibling *_probe.parquet — inputs J_local(this context) @ h_other. If an AV
can predict this row's rollout from those, the Jacobian leaks the target.

Usage (per GPU worker):
  python pretrain/collect_jvp_transport.py \
      --in-shards '/workspace/data/spans_raw/shard_*.parquet' \
      --out-dir /workspace/data/spans_jvp --worker 0 --n-workers 4
"""

import argparse
import glob
import json
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
    ("activation_vector", pa.list_(pa.float32())),   # h42 at p (the test-time input)
    # [16, d] fp16 J_local h, raw bytes: np.frombuffer(b, np.float16).reshape(16, -1)
    ("transported_vectors", pa.binary()),
    ("transport_norms", pa.list_(pa.float32())),     # per-delta pre-cast norms
    ("h42_recompute_cosine", pa.float32()),          # stored-vs-recomputed h42 guard
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
            r["rollout_token_ids"][0], list
        ) else r["rollout_token_ids"]
        seq = list(r["teacher_input_ids"]) + list(roll)
        seqs.append(seq)
        p_list.append(len(r["teacher_input_ids"]) - 1)
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):  # RIGHT pad: harvest positions precede pads
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device), torch.tensor(p_list, device=device)


def jvp_transports(model, ids, mask, p_pos, tangent_vecs):
    """[B, N_OFFSETS, d] forward-mode transports for one batch.

    tangent_vecs: [B, d] — the vector seeded at (L42, p) per row (h42 for
    training rows; another row's h42 for probe rows).
    """
    layers = get_layers(model)
    dtype = next(model.parameters()).dtype
    tangent = tangent_vecs.to(ids.device, dtype)
    rows = torch.arange(ids.shape[0], device=ids.device)

    def run(eps):
        store = {}

        def inject(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            delta = torch.zeros_like(h)
            delta[rows, p_pos] = tangent
            h = h + eps * delta
            if isinstance(output, tuple):
                return (h, *output[1:])
            return h

        def capture(module, inputs, output):
            store["h"] = output[0] if isinstance(output, tuple) else output

        h1 = layers[SRC_LAYER].register_forward_hook(inject)
        h2 = layers[TGT_LAYER].register_forward_hook(capture)
        try:
            model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            h1.remove()
            h2.remove()
        return store["h"]

    eps0 = torch.zeros((), device=ids.device, dtype=dtype)
    tang = torch.ones((), device=ids.device, dtype=dtype)
    primal_out, tangent_out = torch.func.jvp(run, (eps0,), (tang,))

    offs = torch.arange(N_OFFSETS, device=ids.device)
    gather_pos = p_pos[:, None] + offs[None, :]  # [B, 16]
    transported = tangent_out[rows[:, None], gather_pos]  # [B, 16, d]
    return transported.float(), primal_out, gather_pos


def fd_transports(model, ids, mask, p_pos, tangent_vecs, eps_rel=0.05):
    """Central finite differences: needs an fp32 model to be trustworthy."""
    layers = get_layers(model)
    dtype = next(model.parameters()).dtype
    tangent = tangent_vecs.to(ids.device, dtype)
    rows = torch.arange(ids.shape[0], device=ids.device)
    outs = {}

    def run_shift(scale):
        store = {}

        def inject(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[rows, p_pos] = h[rows, p_pos] + scale * tangent
            if isinstance(output, tuple):
                return (h, *output[1:])
            return h

        def capture(module, inputs, output):
            store["h"] = (output[0] if isinstance(output, tuple) else output).detach()

        h1 = layers[SRC_LAYER].register_forward_hook(inject)
        h2 = layers[TGT_LAYER].register_forward_hook(capture)
        try:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            h1.remove()
            h2.remove()
        return store["h"]

    hi = run_shift(eps_rel)
    lo = run_shift(-eps_rel)
    diff = (hi.float() - lo.float()) / (2 * eps_rel)
    offs = torch.arange(N_OFFSETS, device=ids.device)
    gather_pos = p_pos[:, None] + offs[None, :]
    return diff[rows[:, None], gather_pos], gather_pos


def capture_h42(model, ids, mask, p_pos):
    """Recompute h42 at p for the stored-activation consistency guard."""
    layers = get_layers(model)
    store = {}

    def cap(module, inputs, output):
        store["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    h = layers[SRC_LAYER].register_forward_hook(cap)
    try:
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask, use_cache=False)
    finally:
        h.remove()
    rows = torch.arange(ids.shape[0], device=ids.device)
    return store["h"][rows, p_pos].float()


def write_rows(out_path, rows_out):
    cols = {name: [] for name, _ in OUT_SCHEMA_FIELDS}
    for r in rows_out:
        for name, _ in OUT_SCHEMA_FIELDS:
            cols[name].append(r[name])
    arrays = {n: pa.array(cols[n], type=t) for n, t in OUT_SCHEMA_FIELDS}
    table = pa.table(arrays)
    tmp = str(out_path) + ".tmp"
    pq.write_table(table, tmp, row_group_size=1000)
    os.replace(tmp, out_path)
    return table.num_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--in-shards", required=True, help="glob of pass-1 parquets")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--probe-frac", type=float, default=0.02)
    ap.add_argument("--limit-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--selftest", action="store_true",
                    help="compare bf16 JVP vs fp32 central FD on a few rows, then exit")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        import fla  # noqa: F401
        raise SystemExit(
            "flash-linear-attention is installed — its Triton kernels lack "
            "forward-AD rules. Run this collector in a venv without fla."
        )
    except ImportError:
        pass

    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=dtype, attn_implementation="eager"
    ).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    shards = sorted(glob.glob(args.in_shards))
    assert shards, f"no shards match {args.in_shards}"
    my_shards = shards[args.worker :: args.n_workers]
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed + args.worker)

    if args.selftest:
        run_selftest(args, model, tok, pad_id, my_shards[0])
        return

    for shard in my_shards:
        out_path = Path(args.out_dir) / (Path(shard).stem + "_jvp.parquet")
        probe_path = Path(args.out_dir) / (Path(shard).stem + "_jvp_probe.parquet")
        if out_path.exists():
            print(f"[skip] {out_path} exists", flush=True)
            continue
        table = pq.read_table(shard)
        rows = table.to_pylist()
        if args.limit_rows:
            rows = rows[: args.limit_rows]
        rows = [r for r in rows if r["rollout_token_ids"]
                and len(r["rollout_token_ids"][0]) >= N_OFFSETS]
        print(f"[{Path(shard).name}] {len(rows)} rows", flush=True)

        rows_out, probe_out = [], []
        for bi, batch in enumerate(batch_rows(rows, args.batch_size)):
            ids, mask, p_pos = prepare_batch(batch, pad_id, "cuda")
            stored = torch.tensor(
                np.array([r[f"act_L{SRC_LAYER}"] for r in batch], dtype=np.float32)
            ).cuda()
            h42_re = capture_h42(model, ids, mask, p_pos)
            cos = torch.nn.functional.cosine_similarity(h42_re, stored, dim=-1)

            transported, _, _ = jvp_transports(model, ids, mask, p_pos, stored)
            norms = transported.norm(dim=-1)  # [B, 16]

            for i, r in enumerate(batch):
                roll = r["rollout_token_ids"][0]
                rows_out.append({
                    "doc_id": r["doc_id"],
                    "ctx_text": r["ctx_text"],
                    "rollout_text": r["rollouts"][0] if r.get("rollouts") else "",
                    "rollout_token_ids": [int(t) for t in roll],
                    "activation_vector": [float(x) for x in r[f"act_L{SRC_LAYER}"]],
                    "transported_vectors": transported[i].cpu().numpy()
                        .astype(np.float16).tobytes(),
                    "transport_norms": norms[i].cpu().numpy().astype(np.float32)
                        .tolist(),
                    "h42_recompute_cosine": float(cos[i]),
                })

            if args.probe_frac > 0 and rng.random() < args.probe_frac \
                    and len(batch) >= 2:
                # derangement: every row gets a different row's tangent
                perm = (torch.arange(len(batch)) + 1) % len(batch)
                probe_t, _, _ = jvp_transports(
                    model, ids, mask, p_pos, stored[perm]
                )
                for i, r in enumerate(batch):
                    probe_out.append({
                        "doc_id": r["doc_id"],
                        "ctx_text": r["ctx_text"],
                        "rollout_text": r["rollouts"][0] if r.get("rollouts") else "",
                        "rollout_token_ids": [int(t) for t in
                                              r["rollout_token_ids"][0]],
                        "activation_vector": [float(x) for x in
                                              r[f"act_L{SRC_LAYER}"]],
                        "transported_vectors": probe_t[i].cpu().numpy()
                            .astype(np.float16).tobytes(),
                        "transport_norms": probe_t[i].norm(dim=-1).cpu().numpy()
                            .astype(np.float32).tolist(),
                        "h42_recompute_cosine": float(cos[i]),
                    })

            if bi % 20 == 0:
                lo, hi = norms[:, 0].mean(), norms[:, -1].mean()
                print(f"  batch {bi}: |t| d0={lo:.1f} d15={hi:.1f} "
                      f"h42cos={cos.mean():.4f}", flush=True)

        n = write_rows(out_path, rows_out)
        if probe_out:
            write_rows(probe_path, probe_out)
        print(f"[done] {out_path}: {n} rows (+{len(probe_out)} probe)", flush=True)


def disable_tf32():
    """fp32 validation needs REAL fp32: cuDNN conv TF32 defaults ON, and the
    pure-torch DeltaNet fallback uses F.conv1d, so TF32 quietly quantizes
    every fp32 pass to ~1e-3 — drowning finite differences and decorrelating
    forward vs reverse AD."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def vjp_crosscheck(model, ids, mask, p_pos, tangent_vecs, t_jvp,
                   deltas=(0, 3, 8, 15), n_probes=8):
    """Forward-reverse identity <u, J v> == <J^T u, v> per delta (row 0).

    The RHS uses plain reverse-mode autograd through the same graph — an
    estimator-independent ground truth for the jvp path. n_probes random u
    per delta; returns max relative error per delta.
    """
    layers = get_layers(model)
    dtype = next(model.parameters()).dtype
    row = 0
    v = tangent_vecs[row].to(ids.device, dtype)
    store = {}

    def cap42(module, inputs, output):
        t = output[0] if isinstance(output, tuple) else output
        t.requires_grad_(True)  # params are frozen, so this roots the graph
        store["h42"] = t

    def cap62(module, inputs, output):
        store["h62"] = output[0] if isinstance(output, tuple) else output

    h1 = layers[SRC_LAYER].register_forward_hook(cap42)
    h2 = layers[TGT_LAYER].register_forward_hook(cap62)
    errs = []
    try:
        with torch.enable_grad():
            model(input_ids=ids[row : row + 1], attention_mask=mask[row : row + 1],
                  use_cache=False)
            for delta in deltas:
                worst = 0.0
                for probe in range(n_probes):
                    gen = torch.Generator(device="cpu").manual_seed(
                        1000 * delta + probe)
                    u = torch.randn(store["h62"].shape[-1], generator=gen).to(
                        ids.device)
                    tpos = int(p_pos[row]) + delta
                    out_scalar = (store["h62"][0, tpos].float() * u).sum()
                    g = torch.autograd.grad(out_scalar, store["h42"],
                                            retain_graph=True)[0]
                    lhs = float((t_jvp[row, delta].float() * u).sum())
                    rhs = float((g[0, int(p_pos[row])].float() * v.float()).sum())
                    worst = max(worst, abs(lhs - rhs) / (abs(rhs) + 1e-9))
                errs.append(worst)
    finally:
        h1.remove()
        h2.remove()
    return errs


def run_selftest(args, model_bf16, tok, pad_id, shard):
    """bf16 JVP vs fp32 central-FD agreement on a handful of rows."""
    disable_tf32()
    rows = pq.read_table(shard).to_pylist()[:4]
    rows = [r for r in rows if r["rollout_token_ids"]
            and len(r["rollout_token_ids"][0]) >= N_OFFSETS]
    ids, mask, p_pos = prepare_batch(rows, pad_id, "cuda")
    stored = torch.tensor(
        np.array([r[f"act_L{SRC_LAYER}"] for r in rows], dtype=np.float32)
    ).cuda()

    t_bf16, _, _ = jvp_transports(model_bf16, ids, mask, p_pos, stored)
    # bf16 (54G) + fp32 (108G) both fit on a 183G B200; no need to free.
    model32 = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.float32, attn_implementation="eager"
    ).cuda().eval()
    for p in model32.parameters():
        p.requires_grad_(False)
    t_jvp32, _, _ = jvp_transports(model32, ids, mask, p_pos, stored)
    # FD epsilon must stay in the LINEAR regime: through 20 amplifying blocks a
    # 5% perturbation is far into nonlinearity, so use ~1e-3 relative in fp32.
    t_fd_a, _ = fd_transports(model32, ids, mask, p_pos, stored, eps_rel=1e-3)
    t_fd_b, _ = fd_transports(model32, ids, mask, p_pos, stored, eps_rel=5e-4)

    # Independent reverse-mode check: <u, J v> computed forward (dot of u with
    # the jvp tangent) must equal <J^T u, v> computed by backprop.
    vjp_rel_errs = vjp_crosscheck(model32, ids, mask, p_pos, stored, t_jvp32)

    report = {"vjp_identity_rel_err": vjp_rel_errs}
    print("[selftest] vjp identity rel errs:",
          " ".join(f"{x:.2e}" for x in vjp_rel_errs), flush=True)
    for name, a, b in [
        ("jvp_fp32_vs_fd", t_jvp32, t_fd_a),
        ("fd_eps_selfconsistency", t_fd_a, t_fd_b),
        ("jvp_bf16_vs_jvp_fp32", t_bf16, t_jvp32),
    ]:
        cos = torch.nn.functional.cosine_similarity(
            a.flatten(0, 1), b.flatten(0, 1), dim=-1
        )
        report[name] = {
            "cos_mean": float(cos.mean()),
            "cos_min": float(cos.min()),
        }
        print(f"[selftest] {name}: cos mean={cos.mean():.6f} min={cos.min():.6f} "
              f"(1-mean={1 - float(cos.mean()):.2e})", flush=True)
    per_delta = torch.nn.functional.cosine_similarity(
        t_bf16, t_jvp32, dim=-1
    ).mean(dim=0)
    report["bf16_per_delta_cos"] = [float(x) for x in per_delta]
    print("[selftest] bf16 cos by delta:",
          " ".join(f"{x:.3f}" for x in per_delta), flush=True)
    norms = t_jvp32.norm(dim=-1).mean(dim=0)
    report["fp32_per_delta_norm"] = [float(x) for x in norms]
    print("[selftest] |Jv| by delta:",
          " ".join(f"{x:.2f}" for x in norms), flush=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(report, open(Path(args.out_dir) / "selftest.json", "w"), indent=2)
    vjp_ok = max(report["vjp_identity_rel_err"]) < 1e-2
    fd_ok = report["jvp_fp32_vs_fd"]["cos_min"] > 0.99
    bf16_ok = report["jvp_bf16_vs_jvp_fp32"]["cos_min"] > 0.99
    if vjp_ok and bf16_ok:
        verdict = "JVP VALID (vjp identity) + bf16 OK"
    elif vjp_ok:
        verdict = "JVP VALID but USE FP32 (bf16 drift)"
    elif fd_ok:
        verdict = "vjp identity FAILED but FD agrees — investigate"
    else:
        verdict = "INVALID — jvp fails both cross-checks, do not collect"
    print(f"[selftest] VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
