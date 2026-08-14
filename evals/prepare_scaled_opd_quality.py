"""Prepare selected scaled OPD/SFT readouts for the Sonnet quality judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    analysis = json.loads(Path(args.analysis).read_text())
    step = int(analysis["selection"]["selected_opd_step"])
    results = Path(args.results)
    detail = []
    sources = []
    for arm in ("opd", "sft_matched"):
        for feed in ("L62", "L42"):
            path = results / f"{arm}_{step:07d}_{feed}.json"
            data = json.loads(path.read_text())
            detail.extend([
                {**row, "dataset": "fineweb", "arm": arm, "feed": feed}
                for row in data["detail"]
            ])
            sources.append({
                "path": str(path), "arm": arm, "feed": feed,
                "checkpoint": data["checkpoint"],
                "aggregate": data["aggregate"],
            })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "selected_step": step, "sources": sources, "detail": detail,
    }, indent=2))
    print(f"wrote {out}: {len(detail)} rows at update {step}")


if __name__ == "__main__":
    main()
