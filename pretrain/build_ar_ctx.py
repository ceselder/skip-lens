"""Context-prefix AR dataset for the determinacy sweep.

explanation = decode([last N context tokens ending at p] + [on-policy span roll[:k]]),
wrapped in the <summary> critic template. Sweeping N (0, 32, 128, full) traces AR FVE vs
how much context the span is given. Prediction: N=0 (span only) ~28%, N=full ~90% —
proving the 28% is span->activation indeterminacy, not a fit/capacity bug.

Emits {prefix}_train.parquet (AR format: prompt/act/doc_id) and {prefix}_heldout.parquet
(AV format: response/act/doc_id, doc-disjoint) so the AR heldout-FVE loader works."""
import argparse, glob, random, hashlib
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
TR_SCHEMA = pa.schema([("prompt", pa.string()), ("activation_vector", pa.list_(pa.float32())), ("doc_id", pa.string())])
HO_SCHEMA = pa.schema([("response", pa.string()), ("activation_vector", pa.list_(pa.float32())), ("doc_id", pa.string())])


def is_val(d, vf):
    return int(hashlib.md5(str(d).encode()).hexdigest(), 16) % 10000 < int(vf * 10000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-glob", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--ctx-tokens", required=True, help="int number of context tokens, or 'full'")
    ap.add_argument("--n-rows", type=int, default=60000)
    ap.add_argument("--span-min", type=int, default=4)
    ap.add_argument("--span-max", type=int, default=16)
    ap.add_argument("--full-cap", type=int, default=256)
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--summary-token", action="store_true",
                    help="end the AR prompt with the literal <|summary|> token (for --ar-summary-token read-anchor training)")
    a = ap.parse_args()
    tmpl = ("Summary of the following text: <text>{explanation}</text><|summary|>"
            if a.summary_token else TEMPLATE)
    N = a.full_cap if a.ctx_tokens == "full" else int(a.ctx_tokens)
    tok = AutoTokenizer.from_pretrained(a.base_model, trust_remote_code=True)
    rng = random.Random(a.seed)
    files = sorted(glob.glob(a.shards_glob))
    assert files, a.shards_glob
    wtr = pq.ParquetWriter(a.out_prefix + "_train.parquet", TR_SCHEMA)
    wh = pq.ParquetWriter(a.out_prefix + "_heldout.parquet", HO_SCHEMA)
    ntr = nho = seen = 0
    COLS = ["teacher_input_ids", "rollout_token_ids", "activation_vector", "doc_id"]
    for f in files:
        if seen >= a.n_rows:
            break
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            if seen >= a.n_rows:
                break
            t = pf.read_row_group(rg, columns=COLS)
            tis = t.column("teacher_input_ids").to_pylist()
            rolls = t.column("rollout_token_ids").to_pylist()
            docs = t.column("doc_id").to_pylist()
            tr_p, tr_i, ho_r, ho_i = [], [], [], []
            for i in range(len(tis)):
                if seen >= a.n_rows:
                    break
                toks = tis[i]
                roll0 = next((x for x in (rolls[i] or []) if x), None)
                if not toks or not roll0:
                    continue
                k = rng.randint(a.span_min, a.span_max)
                span_ids = [int(x) for x in roll0[:k]]
                ctx_ids = [int(x) for x in (toks[-N:] if N > 0 else [])]
                expl = tok.decode(ctx_ids + span_ids, skip_special_tokens=True).strip()
                if len(expl) < 2:
                    continue
                seen += 1
                if is_val(docs[i], a.val_frac):
                    ho_r.append(expl); ho_i.append(i)
                else:
                    tr_p.append(tmpl.format(explanation=expl)); tr_i.append(i)
            if tr_i:
                sub = t.take(tr_i)
                wtr.write_table(pa.table({
                    "prompt": pa.array(tr_p, pa.string()),
                    "activation_vector": sub.column("activation_vector").cast(pa.list_(pa.float32())),
                    "doc_id": sub.column("doc_id").cast(pa.string())}).cast(TR_SCHEMA))
                ntr += len(tr_i)
            if ho_i:
                sub = t.take(ho_i)
                wh.write_table(pa.table({
                    "response": pa.array(ho_r, pa.string()),
                    "activation_vector": sub.column("activation_vector").cast(pa.list_(pa.float32())),
                    "doc_id": sub.column("doc_id").cast(pa.string())}).cast(HO_SCHEMA))
                nho += len(ho_i)
    wtr.close(); wh.close()
    print(f"CTX={a.ctx_tokens} (N={N}) DONE train={ntr} heldout={nho}", flush=True)


if __name__ == "__main__":
    main()
