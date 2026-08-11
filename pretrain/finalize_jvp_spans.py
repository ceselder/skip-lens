"""Finalize JVP-transported spans into multi-slot AV-SFT format (arm A).

Input: pass-2 shards from pretrain/collect_jvp_transport.py. Each row becomes
one AV training example:
  prompt   = multi-slot actor template with K consecutive <INJECT> markers
  activation_vector = the first K transported vectors, flattened slot-major
                      to K*d fp32 (train_sft --n-slots K reshapes back)
  response = the first K tokens of the ON-POLICY rollout (slot delta reads
             the state at p+delta, which predicts token p+delta+1, so K slots
             support exactly K response tokens)

Guards: rows with h42_recompute_cosine below --min-h42-cos are dropped LOUDLY
(that guard failing en masse = the ae/fl collection-convention bug). Split is
doc-disjoint by hash, same convention as finalize_span_data.py.
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ACTOR_TEMPLATE_MULTI = (
    "You are shown {k} internal activation vectors captured from a language "
    "model as it reads a passage of text. The vectors, enclosed in <concept> "
    "tags, are the model's state at one position transported to {k} "
    "consecutive future positions: the first vector encodes what the model is "
    "about to generate next, the second what it will generate after that, and "
    "so on. Output the text the model most likely produces over these {k} "
    "positions.\n\n<concept>{markers}</concept>")

INJECT_PLACEHOLDER = "<INJECT>"

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
    ap.add_argument("--shards-glob", required=True, help="pass-2 *_jvp.parquet")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--k-slots", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--min-h42-cos", type=float, default=0.99)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model)
    files = sorted(f for f in glob.glob(args.shards_glob)
                   if not f.endswith("_probe.parquet"))
    assert files, f"no shards matched {args.shards_glob}"
    print(f"{len(files)} shards, K={args.k_slots}", flush=True)

    prompt_msgs = [{
        "role": "user",
        "content": ACTOR_TEMPLATE_MULTI.format(
            k=args.k_slots, markers=INJECT_PLACEHOLDER * args.k_slots),
    }]

    writers = {}

    def get_writer(path):
        if path not in writers:
            writers[path] = pq.ParquetWriter(path + ".tmp", OUT_SCHEMA)
        return writers[path]

    n_train = n_val = n_dropped_cos = n_dropped_short = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            rows = t.to_pylist()
            out = {args.out_train: [], args.out_val: []}
            for r in rows:
                if r["h42_recompute_cosine"] < args.min_h42_cos:
                    n_dropped_cos += 1
                    continue
                roll = r["rollout_token_ids"]
                if len(roll) < args.k_slots:
                    n_dropped_short += 1
                    continue
                tv = np.frombuffer(r["transported_vectors"], dtype=np.float16)
                tv = tv.reshape(16, -1)[: args.k_slots].astype(np.float32)
                resp = tok.decode(roll[: args.k_slots], skip_special_tokens=True)
                if len(resp.strip()) < 2:
                    n_dropped_short += 1
                    continue
                dest = args.out_val if is_val(r["doc_id"], args.val_frac) \
                    else args.out_train
                out[dest].append({
                    "prompt": prompt_msgs,
                    "response": resp,
                    "activation_vector": tv.reshape(-1).tolist(),
                    "ctx_text": r["ctx_text"],
                    "doc_id": r["doc_id"],
                    "span_len": args.k_slots,
                })
            for path, batch in out.items():
                if not batch:
                    continue
                cols = {name: [b[name] for b in batch]
                        for name in OUT_SCHEMA.names}
                get_writer(path).write_table(pa.table(
                    {n: pa.array(cols[n], type=OUT_SCHEMA.field(n).type)
                     for n in OUT_SCHEMA.names}))
                if path == args.out_train:
                    n_train += len(batch)
                else:
                    n_val += len(batch)
        print(f"  {os.path.basename(f)}: train={n_train} val={n_val} "
              f"dropped(cos)={n_dropped_cos} dropped(short)={n_dropped_short}",
              flush=True)

    for path, w in writers.items():
        w.close()
        os.replace(path + ".tmp", path)
    frac_dropped = n_dropped_cos / max(1, n_train + n_val + n_dropped_cos)
    meta = {"k_slots": args.k_slots, "train_rows": n_train, "val_rows": n_val,
            "dropped_low_cos": n_dropped_cos, "dropped_short": n_dropped_short}
    json.dump(meta, open(args.out_train + ".meta.json", "w"), indent=2)
    print(json.dumps(meta), flush=True)

    # Sidecar (nla_meta.yaml): the template stores K literal {injection_char}
    # placeholders so load_nla_config's .format() re-expands the marker run;
    # compute_canonical_neighbors handles runs (flank neighbors).
    import datetime

    import yaml

    from nla.datagen.injection_tokens import find_injection_token
    from nla.schema import compute_canonical_neighbors

    inj_char, inj_id = find_injection_token(tok)
    sidecar_template = ACTOR_TEMPLATE_MULTI.format(
        k=args.k_slots, markers="{injection_char}" * args.k_slots)
    left, right = compute_canonical_neighbors(tok, sidecar_template, inj_char, inj_id)
    d_model = len(np.frombuffer(
        pq.read_table(files[0], columns=["transported_vectors"])
        .column("transported_vectors")[0].as_py(), dtype=np.float16)) // 16
    sidecar = {
        "dataset_id": f"jvpspan_multislot_{args.base_model.split('/')[-1]}"
                      f"_L42to62_k{args.k_slots}",
        "stage": "av_sft", "row_count": n_train, "kind": "nla_dataset",
        "schema_version": 1, "keep_debug_metadata": True,
        "n_slots": args.k_slots,
        "extraction": {"base_model": args.base_model, "d_model": d_model,
                       "layer_index": 62, "source_layer": 42,
                       "transport": "local_jvp_L42_to_L62", "norm": "none"},
        "tokens": {"injection_char": inj_char, "injection_token_id": inj_id,
                   "injection_left_neighbor_id": left,
                   "injection_right_neighbor_id": right,
                   "critic_suffix_ids": None},
        "prompt_templates": {"actor": sidecar_template},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": "pretrain.finalize_jvp_spans",
    }
    for o in (args.out_train, args.out_val):
        with open(o + ".nla_meta.yaml", "w") as fh:
            yaml.safe_dump(sidecar, fh, sort_keys=False, allow_unicode=True)
    print("wrote sidecars", flush=True)
    if frac_dropped > 0.02:
        print(f"WARNING: {frac_dropped:.1%} rows dropped on the h42 recompute "
              f"guard — investigate collection-convention mismatch!", flush=True)


if __name__ == "__main__":
    main()
