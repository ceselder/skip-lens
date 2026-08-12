"""Build a PASTLENS AV dataset (Karvonen activation-oracles 'past_lens', adapted to our skip-lens).

Target = the k tokens immediately BEFORE the activation position p (the context the model just
READ), forward order, NOT including p:  ids[p-k : p],  k ~ U[span_min, span_max].
Prompt is PAST-framed. The past is real/deterministic (no on-policy needed). Built from the SAME
spans_onpolicy activations/positions as the futurelens sweep (teacher_input_ids = ids[:p+1]),
so pastlens vs futurelens is a controlled same-activation comparison.

Streaming (row-group by row-group), doc-disjoint val split by stable hash of doc_id.
Train with `nla.train_sft --mode av` exactly like the futurelens."""
import argparse, glob, json, datetime, random, hashlib
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

PAST_ACTOR_TEMPLATE = (
    "You are shown an internal activation vector captured from a language model "
    "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
    "taken at one position and encodes the context the model has just read. "
    "Output the text that immediately preceded this point.\n\n"
    "<concept>{injection_char}</concept>")

PS = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
OUT_SCHEMA = pa.schema([
    ("prompt", PS), ("response", pa.string()),
    ("activation_vector", pa.list_(pa.float32())), ("ctx_text", pa.string()),
    ("doc_id", pa.string()), ("span_len", pa.int32()),
])


def is_val(doc_id, val_frac):
    return int(hashlib.md5(str(doc_id).encode()).hexdigest(), 16) % 10000 < int(val_frac * 10000)


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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    rng = random.Random(args.seed)
    m = json.loads(Path(args.meta).read_text())
    past_prompt = [{"role": "user", "content": PAST_ACTOR_TEMPLATE.format(injection_char=m["inj_char"])}]
    files = sorted(glob.glob(args.shards_glob))
    assert files, f"no shards matched {args.shards_glob}"
    print(f"{len(files)} shards; PAST target (ids[p-k:p], forward, k~U[{args.span_min},{args.span_max}])", flush=True)

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    w_tr = pq.ParquetWriter(args.out_train, OUT_SCHEMA)
    w_va = pq.ParquetWriter(args.out_val, OUT_SCHEMA)
    n_tr = n_va = n_skip = 0
    span_sum = span_n = 0
    COLS = ["teacher_input_ids", "activation_vector", "ctx_text", "doc_id"]

    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg, columns=COLS)
            tis = t.column("teacher_input_ids").to_pylist()
            docids = t.column("doc_id").to_pylist()
            ks, sliced, keep = [], [], []
            for i, toks in enumerate(tis):
                # toks = ids[:p+1]; the k tokens before p = toks[len-1-k : len-1] (exclude p)
                if not toks or len(toks) < args.span_min + 1:
                    n_skip += 1; continue
                k = rng.randint(args.span_min, min(args.span_max, len(toks) - 1))
                past = [int(x) for x in toks[len(toks) - 1 - k: len(toks) - 1]]
                ks.append(len(past)); sliced.append(past); keep.append(i)
            resps = tok.batch_decode(sliced, skip_special_tokens=True)
            tr_idx, va_idx, tr_r, va_r, tr_l, va_l = [], [], [], [], [], []
            for j, i in enumerate(keep):
                r = resps[j].strip()
                if len(r) < 2:
                    n_skip += 1; continue
                span_sum += ks[j]; span_n += 1
                if is_val(docids[i], args.val_frac):
                    va_idx.append(i); va_r.append(r); va_l.append(ks[j])
                else:
                    tr_idx.append(i); tr_r.append(r); tr_l.append(ks[j])

            def emit(writer, idx, resp, slen):
                if not idx:
                    return 0
                sub = t.take(idx)
                chunk = pa.table({
                    "prompt": pa.array([past_prompt] * len(idx), type=PS),
                    "response": pa.array(resp, type=pa.string()),
                    "activation_vector": sub.column("activation_vector").cast(pa.list_(pa.float32())),
                    "ctx_text": sub.column("ctx_text").cast(pa.string()),
                    "doc_id": sub.column("doc_id").cast(pa.string()),
                    "span_len": pa.array(slen, type=pa.int32()),
                }).cast(OUT_SCHEMA)
                writer.write_table(chunk)
                return len(idx)
            n_tr += emit(w_tr, tr_idx, tr_r, tr_l)
            n_va += emit(w_va, va_idx, va_r, va_l)
        print(f"  {Path(f).name}: train={n_tr} val={n_va}", flush=True)
    w_tr.close(); w_va.close()
    print(f"PASTLENS DONE train={n_tr} val={n_va} skip={n_skip} span_mean={span_sum/max(1,span_n):.2f}", flush=True)

    sidecar = {
        "dataset_id": f"pastlens_{args.base_model.split('/')[-1]}_L{m['layer']}",
        "stage": "av_sft", "row_count": n_tr, "kind": "nla_dataset",
        "schema_version": 1, "keep_debug_metadata": True,
        "span_len_range": [args.span_min, args.span_max],
        "extraction": {"base_model": args.base_model, "d_model": m["d_model"],
                       "layer_index": m["layer"], "norm": "none"},
        "tokens": {"injection_char": m["inj_char"], "injection_token_id": m["inj_id"],
                   "injection_left_neighbor_id": m["left"], "injection_right_neighbor_id": m["right"],
                   "critic_suffix_ids": None},
        "prompt_templates": {"actor": PAST_ACTOR_TEMPLATE},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": "pretrain.build_pastlens",
    }
    for o in (args.out_train, args.out_val):
        Path(o + ".nla_meta.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False, allow_unicode=True))
    print("wrote sidecars", flush=True)


if __name__ == "__main__":
    main()
