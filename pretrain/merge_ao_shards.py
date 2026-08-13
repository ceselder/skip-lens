"""Merge non-overlapping ``collect_ao_data`` Parquet shards and metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = [Path(path) for path in args.inputs]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing collection shards: {missing}")

    tables = [pq.read_table(path) for path in paths]
    schema = tables[0].schema
    for path, table in zip(paths[1:], tables[1:]):
        if table.schema != schema:
            raise ValueError(f"schema mismatch in {path}")
    merged = pa.concat_tables(tables)
    doc_ids = merged.column("doc_id").to_pylist()
    if len(set(doc_ids)) != sum(len(set(t.column("doc_id").to_pylist())) for t in tables):
        raise ValueError("document IDs overlap across collection shards")

    metas = [json.loads(Path(str(path) + ".meta.json").read_text()) for path in paths]
    ignored = {"rows"}
    reference = {key: value for key, value in metas[0].items() if key not in ignored}
    for path, meta in zip(paths[1:], metas[1:]):
        comparable = {key: value for key, value in meta.items() if key not in ignored}
        if comparable != reference:
            raise ValueError(f"collection metadata mismatch in {path}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, out, row_group_size=2000)
    output_meta = dict(metas[0])
    output_meta["rows"] = merged.num_rows
    output_meta["merged_shards"] = [str(path) for path in paths]
    Path(str(out) + ".meta.json").write_text(json.dumps(output_meta, indent=2) + "\n")
    print(
        f"wrote {out}: {merged.num_rows} rows, "
        f"{len(set(doc_ids))} documents from {len(paths)} shards"
    )


if __name__ == "__main__":
    main()
