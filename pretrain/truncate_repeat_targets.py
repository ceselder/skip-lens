"""Create a matched repeat-control dataset with shorter token targets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    args = ap.parse_args()
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")

    src = Path(args.input)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    pf = pq.ParquetFile(src)
    writer = None
    rows_written = 0
    try:
        for rg_idx in range(pf.num_row_groups):
            rows = pf.read_row_group(rg_idx).to_pylist()
            for row in rows:
                target = list(map(int, row["target_ids"][:args.tokens]))
                if len(target) != args.tokens:
                    raise ValueError(
                        f"row {rows_written} has only {len(target)} target tokens"
                    )
                row["target_ids"] = target
                if "continuation_ids" in row:
                    row["continuation_ids"] = list(
                        map(int, row["continuation_ids"][:args.tokens])
                    )
                row["response"] = tok.decode(target, skip_special_tokens=False)
                if tok.encode(row["response"], add_special_tokens=False) != target:
                    raise ValueError(f"target decode/encode mismatch at row {rows_written}")
                if "span_tokens" in row:
                    row["span_tokens"] = args.tokens
            table = pa.Table.from_pylist(rows, schema=pf.schema_arrow)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
            rows_written += len(rows)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError(f"no rows in {src}")

    for suffix in (".nla_meta.yaml", ".manifest.json"):
        sidecar = Path(str(src) + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, Path(str(out) + suffix))
    manifest_path = Path(str(out) + ".target_transform.json")
    manifest_path.write_text(json.dumps({
        "source": str(src),
        "out": str(out),
        "rows": rows_written,
        "target_tokens": args.tokens,
        "base_ckpt": args.base_ckpt,
    }, indent=2) + "\n")
    print(json.dumps({"out": str(out), "rows": rows_written,
                      "target_tokens": args.tokens}))


if __name__ == "__main__":
    main()
