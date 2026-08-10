"""Stream-merge repeat-control Parquet shards without materialising vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    rows = 0
    schema = None
    try:
        for input_path in args.inputs:
            parquet = pq.ParquetFile(input_path)
            if schema is None:
                schema = parquet.schema_arrow
                writer = pq.ParquetWriter(out, schema)
            elif parquet.schema_arrow != schema:
                raise ValueError(f"schema mismatch in {input_path}")
            for row_group in range(parquet.num_row_groups):
                table = parquet.read_row_group(row_group)
                writer.write_table(table)
                rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("no input shards")

    first_sidecar = Path(args.inputs[0] + ".nla_meta.yaml")
    if first_sidecar.exists():
        sidecar = yaml.safe_load(first_sidecar.read_text())
        sidecar["row_count"] = rows
        sidecar["merged_from"] = list(args.inputs)
        Path(str(out) + ".nla_meta.yaml").write_text(
            yaml.safe_dump(sidecar, sort_keys=False, allow_unicode=True)
        )
    manifest = {
        "rows": rows,
        "shards": list(args.inputs),
        "source_manifests": [
            json.loads(Path(path + ".manifest.json").read_text())
            for path in args.inputs
            if Path(path + ".manifest.json").exists()
        ],
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"out": str(out), "rows": rows, "shards": len(args.inputs)}))


if __name__ == "__main__":
    main()
