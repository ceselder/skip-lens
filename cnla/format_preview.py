"""Turn a collect_ao_data parquet (with a `rollouts` list column of K sampled
continuations) into the compositional-NLA 4-bullet format + a human-readable preview.

Adds a `bullets_text` column:  "* c1\n* c2\n* c3\n* c4"  (each cᵢ is one sampled
completion, internal newlines flattened so the `*` markers stay parseable — the RL
reward parses bullets by splitting on the leading "* "). Writes:
  <out>.parquet        the input rows + bullets_text
  <out>_preview.txt    first --n rows rendered for eyeballing (ctx + the 4 bullets)
  <out>_preview.jsonl  same, structured
"""
import argparse, json, re
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def clean(c: str) -> str:
    # one bullet = one completion; flatten internal newlines + collapse whitespace so
    # the "* " line-markers remain the only bullet delimiters.
    return re.sub(r"\s+", " ", (c or "").replace("\n", " ")).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True, help="output stem (no extension)")
    ap.add_argument("--n-preview", type=int, default=30)
    args = ap.parse_args()

    t = pq.read_table(args.inp)
    rows = t.to_pylist()
    bullets_text = []
    for r in rows:
        conts = [clean(c) for c in (r.get("rollouts") or []) if clean(c)]
        bullets_text.append("\n".join(f"* {c}" for c in conts))
    out_tbl = t.append_column("bullets_text", pa.array(bullets_text, type=pa.string()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_tbl, args.out + ".parquet", row_group_size=2000)

    # readable previews
    prev = []
    with open(args.out + "_preview.txt", "w") as f:
        f.write(f"CNLA 4-bullet dataset preview — {out_tbl.num_rows} rows total\n")
        f.write("=" * 88 + "\n\n")
        for i, r in enumerate(rows[: args.n_preview]):
            ctx = clean(r.get("ctx_text", ""))
            ent = r.get("next_token_entropy")
            f.write(f"[{i}] doc={r.get('doc_id','?')}  next-token entropy={ent:.3f}\n" if ent == ent
                    else f"[{i}] doc={r.get('doc_id','?')}\n")
            f.write(f"    context tail: …{ctx[-180:]}\n")
            f.write(f"    4 sampled completions @T=1.0 (penultimate L62 activation stored):\n")
            for line in bullets_text[i].split("\n"):
                f.write(f"      {line}\n")
            f.write("\n")
            prev.append({"idx": i, "doc_id": r.get("doc_id"), "ctx_tail": ctx[-180:],
                         "entropy": ent, "bullets_text": bullets_text[i]})
    Path(args.out + "_preview.jsonl").write_text("\n".join(json.dumps(p) for p in prev))
    print(f"wrote {args.out}.parquet ({out_tbl.num_rows} rows), preview txt+jsonl "
          f"({min(args.n_preview, len(rows))} rows)")


if __name__ == "__main__":
    main()
