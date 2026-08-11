"""One-pass FineFineWeb materialization for the offset-lens pipeline.

Streams the corpus head once and writes TWO disjoint parquets:
  corpus:  rows [0, n_corpus)            -> span collection (pass 1)
  pool:    rows [n_corpus, +n_pool)      -> Jacobian-fit prompts (long docs)

Disjointness matters: the averaged Jacobian must not be fitted on the same
documents the spans are collected from.
"""
import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def write(rows, path):
    table = pa.table({"text": pa.array(rows, type=pa.string())})
    pq.write_table(table, path, row_group_size=5000)
    print(f"wrote {path}: {table.num_rows} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-out", default="/workspace/data/ffw_corpus_100k.parquet")
    ap.add_argument("--pool-out", default="/workspace/data/ffw_fit_pool.parquet")
    ap.add_argument("--n-corpus", type=int, default=100_000)
    ap.add_argument("--n-pool", type=int, default=10_000)
    ap.add_argument("--min-chars", type=int, default=200)
    args = ap.parse_args()

    ds = load_dataset("m-a-p/FineFineWeb", split="train", streaming=True)
    corpus, pool = [], []
    for row in ds:
        text = row.get("text") or ""
        if len(text) < args.min_chars:
            continue
        if len(corpus) < args.n_corpus:
            corpus.append(text)
        elif len(pool) < args.n_pool:
            pool.append(text)
        else:
            break
        n = len(corpus) + len(pool)
        if n % 20_000 == 0:
            print(f"  {n} rows...", flush=True)

    write(corpus, args.corpus_out)
    write(pool, args.pool_out)


if __name__ == "__main__":
    main()
