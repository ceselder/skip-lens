"""Select the saved SFT checkpoint with the best held-out metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--metric", default="heldout_loss")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    candidates = []
    for line in Path(args.metrics).read_text().splitlines():
        row = json.loads(line)
        if args.metric not in row:
            continue
        update = int(row["step"]) + 1
        checkpoint = Path(args.checkpoint_dir) / f"iter_{update:07d}"
        if checkpoint.is_dir():
            candidates.append((float(row[args.metric]), update, checkpoint, row))
    if not candidates:
        raise ValueError(
            f"no saved checkpoint has metric {args.metric!r} in {args.metrics}"
        )
    value, update, checkpoint, row = min(candidates)
    result = {
        "metric": args.metric,
        "value": value,
        "update": update,
        "checkpoint": str(checkpoint),
        "metrics": row,
        "n_candidates": len(candidates),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
