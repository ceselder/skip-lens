"""Create a matched repeat-control dataset with shorter token targets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


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
    pf = pq.ParquetFile(src)
    writer = None
    rows_written = 0
    try:
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx)
            targets = [list(map(int, x[:args.tokens]))
                       for x in table["target_ids"].to_pylist()]
            continuations = [list(map(int, x[:args.tokens]))
                             for x in table["continuation_ids"].to_pylist()]
            original_responses = table["response"].to_pylist()
            responses = []
            for row_idx, target in enumerate(targets):
                if len(target) != args.tokens:
                    raise ValueError(
                        f"row {rows_written + row_idx} has only {len(target)} target tokens"
                    )
                if len(continuations[row_idx]) != args.tokens:
                    raise ValueError(
                        f"row {rows_written + row_idx} has a short continuation"
                    )
                original = original_responses[row_idx]
                words = original.split()
                if len(words) < args.tokens:
                    raise ValueError(
                        f"row {rows_written + row_idx} has only {len(words)} response words"
                    )
                # Every generator word was pre-filtered to exactly one token in
                # both bare and space-prefixed forms. Preserve the source's
                # leading whitespace and take the first N words.
                leading = original[:len(original) - len(original.lstrip())]
                response = leading + " ".join(words[:args.tokens])
                responses.append(response)
            replacements = {
                "target_ids": pa.array(
                    targets, type=pf.schema_arrow.field("target_ids").type),
                "continuation_ids": pa.array(
                    continuations, type=pf.schema_arrow.field("continuation_ids").type),
                "response": pa.array(
                    responses, type=pf.schema_arrow.field("response").type),
                "span_tokens": pa.array(
                    [args.tokens] * table.num_rows,
                    type=pf.schema_arrow.field("span_tokens").type),
            }
            for name, values in replacements.items():
                idx = table.schema.get_field_index(name)
                table = table.set_column(idx, name, values)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
            rows_written += table.num_rows
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
