"""Merge per-checkpoint workspace readouts and retain training-step identity."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = []
    checkpoints = []
    for input_path in args.inputs:
        path = Path(input_path)
        match = re.search(r"step_(\d+)", path.stem)
        if not match:
            raise ValueError(f"cannot parse step from {path}")
        step = int(match.group(1))
        data = json.loads(path.read_text())
        checkpoint = data["meta"]["checkpoint"]
        checkpoints.append({"step": step, "checkpoint": checkpoint, "input": str(path)})
        records.extend({**record, "training_step": step} for record in data["records"])
    out = {
        "meta": {
            "trajectory": True,
            "checkpoints": sorted(checkpoints, key=lambda x: x["step"]),
            "layers": [42, 62],
            "modes": ["raw"],
        },
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}: {len(checkpoints)} checkpoints, {len(records)} records")


if __name__ == "__main__":
    main()
