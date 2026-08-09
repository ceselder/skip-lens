"""Merge OPD evaluation details while retaining experimental condition labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    detail = []
    sources = []
    for input_path in args.inputs:
        path = Path(input_path)
        data = json.loads(path.read_text())
        stem = path.stem
        # Stage-3 files use {dataset}_{arm}_{feed}_eval.json.
        parts = stem.removesuffix("_eval").split("_")
        dataset = parts[0]
        feed = parts[-1]
        arm = "_".join(parts[1:-1])
        rows = [
            {**row, "dataset": dataset, "arm": arm, "feed": feed}
            for row in data["detail"]
        ]
        detail.extend(rows)
        sources.append({
            "path": str(path),
            "dataset": dataset,
            "arm": arm,
            "feed": feed,
            "checkpoint": data.get("checkpoint"),
            "aggregate": data.get("aggregate"),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sources": sources, "detail": detail}, indent=2))
    print(f"wrote {out}: {len(detail)} rows from {len(sources)} evaluations")


if __name__ == "__main__":
    main()
