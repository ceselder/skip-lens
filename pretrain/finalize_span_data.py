"""Finalize REAL-span collection into AV-SFT format (STREAMING, memory-bounded).

Target = the ACTUAL corpus continuation: response = decode(continuation_ids[:k]),
k ~ U[span_min, span_max] per row. Forward-only collection (--no-rollouts) leaves
`rollouts` empty but fills `continuation_ids` (ids[p+1 : p+1+rollout_len]).

Streams shard-by-shard, row-group by row-group (never loads the full set into RAM),
routes each doc to train/val by a stable hash of doc_id (doc-disjoint split). Writes
AV train/val + sidecar. Run scripts/build_ar_big.py afterwards for the AR form."""
import argparse, glob, json, datetime, random, hashlib
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ACTOR_TEMPLATE = (
    "You are shown an internal activation vector captured from a language model "
    "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
    "taken at one position and encodes what the model is about to generate next. "
    "Output the text the model most likely produces immediately after this point.\n\n"
    "<concept>{injection_char}</concept>")

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
    ap.add_argument("--shards-glob", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--span-min", type=int, default=4)
    ap.add_argument("--span-max", type=int, default=16)
    ap.add_argument("--min-chars", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--span-source", choices=["rollout", "continuation"], default="rollout",
                    help="rollout = the model's OWN generated tokens (ON-POLICY, rollout_token_ids[0]); "
                         "continuation = the real corpus next tokens (off-policy, continuation_ids)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    rng = random.Random(args.seed)
    files = sorted(glob.glob(args.shards_glob))
    assert files, f"no shards matched {args.shards_glob}"
    print(f"{len(files)} shards", flush=True)

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    w_tr = pq.ParquetWriter(args.out_train, OUT_SCHEMA)
    w_va = pq.ParquetWriter(args.out_val, OUT_SCHEMA)
    n_tr = n_va = n_empty = n_nocont = 0
    span_sum = span_n = 0
    SRC_COL = "rollout_token_ids" if args.span_source == "rollout" else "continuation_ids"
    COLS = ["prompt", SRC_COL, "activation_vector", "ctx_text", "doc_id"]
    print(f"span source: {args.span_source} ({SRC_COL})", flush=True)

    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=COLS)
            raw = t.column(SRC_COL).to_pylist()
            docids = t.column("doc_id").to_pylist()
            # decode span[:k] per row (batched decode per row-group)
            ks, sliced, keep = [], [], []
            for i, c in enumerate(raw):
                # rollout_token_ids is list<list<int>> (per-rollout) -> take first non-empty rollout
                toks = next((x for x in (c or []) if x), None) if args.span_source == "rollout" else c
                if not toks:
                    n_nocont += 1; continue
                k = rng.randint(args.span_min, args.span_max)
                ks.append(len(toks[:k])); sliced.append([int(x) for x in toks[:k]]); keep.append(i)
            resps = tok.batch_decode(sliced, skip_special_tokens=True)
            tr_idx, va_idx, tr_resp, va_resp, tr_len, va_len = [], [], [], [], [], []
            for j, i in enumerate(keep):
                r = resps[j].strip()
                if len(r) < args.min_chars:
                    n_empty += 1; continue
                span_sum += ks[j]; span_n += 1
                if is_val(docids[i], args.val_frac):
                    va_idx.append(i); va_resp.append(r); va_len.append(ks[j])
                else:
                    tr_idx.append(i); tr_resp.append(r); tr_len.append(ks[j])

            def emit(writer, idx, resp, slen):
                if not idx:
                    return 0
                sub = t.take(idx)
                chunk = pa.table({
                    "prompt": sub.column("prompt").cast(PS),
                    "response": pa.array(resp, type=pa.string()),
                    "activation_vector": sub.column("activation_vector").cast(pa.list_(pa.float32())),
                    "ctx_text": sub.column("ctx_text").cast(pa.string()),
                    "doc_id": sub.column("doc_id").cast(pa.string()),
                    "span_len": pa.array(slen, type=pa.int32()),
                }).cast(OUT_SCHEMA)
                writer.write_table(chunk)
                return len(idx)
            n_tr += emit(w_tr, tr_idx, tr_resp, tr_len)
            n_va += emit(w_va, va_idx, va_resp, va_len)
        print(f"  {Path(f).name} done: train={n_tr} val={n_va}", flush=True)
    w_tr.close(); w_va.close()
    print(f"DONE train={n_tr} val={n_va} no_cont={n_nocont} empty={n_empty} "
          f"span_len_mean={span_sum/max(1,span_n):.2f}", flush=True)

    m = json.loads(Path(args.meta).read_text())
    sidecar = {
        "dataset_id": f"realspan_fl_{args.base_model.split('/')[-1]}_L{m['layer']}",
        "stage": "av_sft", "row_count": n_tr, "kind": "nla_dataset",
        "schema_version": 1, "keep_debug_metadata": True,
        "span_len_range": [args.span_min, args.span_max],
        "extraction": {"base_model": args.base_model, "d_model": m["d_model"],
                       "layer_index": m["layer"], "norm": "none"},
        "tokens": {"injection_char": m["inj_char"], "injection_token_id": m["inj_id"],
                   "injection_left_neighbor_id": m["left"], "injection_right_neighbor_id": m["right"],
                   "critic_suffix_ids": None},
        "prompt_templates": {"actor": ACTOR_TEMPLATE},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": "pretrain.finalize_span_data",
    }
    import yaml
    for o in (args.out_train, args.out_val):
        Path(o + ".nla_meta.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False, allow_unicode=True))
    print("wrote sidecars", flush=True)


if __name__ == "__main__":
    main()
