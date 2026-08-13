"""Build the two-J (arm E) training parquet — no new collection needed.

Training slots are  Jbar^(d)_{62->63} @ h62[p], and h62[p] is already stored in
the pass-1 shards as `act_L62` (note: the raw shards' `activation_vector` column
is ALSO act_L62 by the arm-C convention, but we read act_L62 explicitly). So
this is a pure matmul over existing data.

Response = the first K tokens of the ON-POLICY rollout, matching the eval-time
slot count. Split is doc-disjoint by the same md5 rule as the other finalizers,
and the decoded response must round-trip to the rollout ids (BPE re-merge guard).
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from pretrain.finalize_jvp_spans import ACTOR_TEMPLATE_MULTI, INJECT_PLACEHOLDER

PS = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
OUT_SCHEMA = pa.schema([
    ("prompt", PS),
    ("response", pa.string()),
    ("activation_vector", pa.list_(pa.float32())),
    ("ctx_text", pa.string()),
    ("doc_id", pa.string()),
    ("span_len", pa.int32()),
])


def is_val(doc_id, val_frac):
    h = int(hashlib.md5(str(doc_id).encode()).hexdigest(), 16) % 10000
    return h < int(val_frac * 10000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-glob", required=True)
    ap.add_argument("--jbar-dir", required=True)
    ap.add_argument("--src-layer", type=int, default=62)
    ap.add_argument("--tgt-layer", type=int, default=63)
    ap.add_argument("--k-slots", type=int, default=8)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--val-frac", type=float, default=0.03)
    args = ap.parse_args()
    K = args.k_slots
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base_model)
    Jb = [torch.from_numpy(np.load(
        f"{args.jbar_dir}/Jbar_L{args.src_layer}_to_L{args.tgt_layer}_off{d}.npy")
        ).float().to(dev) for d in range(K)]
    print(f"loaded {K} Jbar_L{args.src_layer}->L{args.tgt_layer} matrices "
          f"(|J| d0={float(Jb[0].norm()):.2f} d7={float(Jb[-1].norm()):.2f})", flush=True)

    prompt_msgs = [{"role": "user", "content": ACTOR_TEMPLATE_MULTI.format(
        k=K, markers=INJECT_PLACEHOLDER * K)}]
    files = sorted(glob.glob(args.raw_glob))
    assert files, f"no shards matched {args.raw_glob}"

    writers, n = {}, {"train": 0, "val": 0, "short": 0, "roundtrip": 0}

    def wr(path):
        if path not in writers:
            writers[path] = pq.ParquetWriter(path + ".tmp", OUT_SCHEMA)
        return writers[path]

    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            rows = pf.read_row_group(rg, columns=[
                "act_L62", "rollout_token_ids", "ctx_text", "doc_id"]).to_pylist()
            keep, h62 = [], []
            for r in rows:
                roll = r["rollout_token_ids"][0] if r["rollout_token_ids"] else []
                if len(roll) < K:
                    n["short"] += 1
                    continue
                resp = tok.decode(roll[:K], skip_special_tokens=True)
                if len(resp.strip()) < 2:
                    n["short"] += 1
                    continue
                if tok.encode(resp, add_special_tokens=False) != [int(x) for x in roll[:K]]:
                    n["roundtrip"] += 1
                    continue
                keep.append((r, resp))
                h62.append(r["act_L62"])
            if not keep:
                continue
            H = torch.tensor(np.array(h62, dtype=np.float32), device=dev)   # [B, d]
            slots = torch.stack([(Jb[d] @ H.T).T for d in range(K)], dim=1)  # [B, K, d]
            out = {args.out_train: [], args.out_val: []}
            for i, (r, resp) in enumerate(keep):
                dest = args.out_val if is_val(r["doc_id"], args.val_frac) else args.out_train
                out[dest].append({
                    "prompt": prompt_msgs, "response": resp,
                    "activation_vector": slots[i].reshape(-1).cpu().numpy().tolist(),
                    "ctx_text": r["ctx_text"], "doc_id": r["doc_id"],
                    "span_len": K,
                })
            for path, batch in out.items():
                if not batch:
                    continue
                wr(path).write_table(pa.table({
                    nm: pa.array([b[nm] for b in batch], type=OUT_SCHEMA.field(nm).type)
                    for nm in OUT_SCHEMA.names}))
                n["train" if path == args.out_train else "val"] += len(batch)
        print(f"  {os.path.basename(f)}: train={n['train']} val={n['val']} "
              f"short={n['short']} roundtrip={n['roundtrip']}", flush=True)

    for path, w in writers.items():
        w.close()
        os.replace(path + ".tmp", path)

    import datetime

    import yaml

    from nla.datagen.injection_tokens import find_injection_token
    from nla.schema import compute_canonical_neighbors
    inj_char, inj_id = find_injection_token(tok)
    tmpl = ACTOR_TEMPLATE_MULTI.format(k=K, markers="{injection_char}" * K)
    left, right = compute_canonical_neighbors(tok, tmpl, inj_char, inj_id)
    sidecar = {
        "dataset_id": f"twoJ_multislot_L{args.src_layer}to{args.tgt_layer}_k{K}",
        "stage": "av_sft", "row_count": n["train"], "kind": "nla_dataset",
        "schema_version": 1, "keep_debug_metadata": True, "n_slots": K,
        "extraction": {"base_model": args.base_model, "d_model": int(Jb[0].shape[0]),
                       "layer_index": args.tgt_layer, "source_layer": args.src_layer,
                       "transport": f"averaged_perOffset_L{args.src_layer}_to_L{args.tgt_layer}",
                       "norm": "none"},
        "tokens": {"injection_char": inj_char, "injection_token_id": inj_id,
                   "injection_left_neighbor_id": left,
                   "injection_right_neighbor_id": right, "critic_suffix_ids": None},
        "prompt_templates": {"actor": tmpl},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": "pretrain.build_twoJ_train",
    }
    for o in (args.out_train, args.out_val):
        with open(o + ".nla_meta.yaml", "w") as fh:
            yaml.safe_dump(sidecar, fh, sort_keys=False, allow_unicode=True)
    json.dump(n, open(args.out_train + ".meta.json", "w"), indent=2)
    print(json.dumps(n), flush=True)


if __name__ == "__main__":
    main()
