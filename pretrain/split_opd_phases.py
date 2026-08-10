"""Create deterministic, row-disjoint token-matched OPD training phases."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-warm", required=True)
    parser.add_argument("--out-stage2", required=True)
    parser.add_argument("--rows-per-phase", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    table = pq.read_table(args.input)
    needed = 2 * args.rows_per_phase
    if table.num_rows < needed:
        raise ValueError(f"need {needed} rows, found {table.num_rows}")
    order = np.random.default_rng(args.seed).permutation(table.num_rows)[:needed]
    warm_idx = order[:args.rows_per_phase]
    stage2_idx = order[args.rows_per_phase:]
    assert not set(warm_idx.tolist()) & set(stage2_idx.tolist())

    source_sidecar = Path(args.input + ".nla_meta.yaml")
    for path, indices in (
        (args.out_warm, warm_idx),
        (args.out_stage2, stage2_idx),
    ):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.take(pa.array(indices)), out, row_group_size=2000)
        if source_sidecar.exists():
            shutil.copy2(source_sidecar, Path(path + ".nla_meta.yaml"))

    print(
        f"warm={len(warm_idx)} stage2={len(stage2_idx)} "
        f"overlap={len(set(warm_idx.tolist()) & set(stage2_idx.tolist()))}"
    )


if __name__ == "__main__":
    main()
