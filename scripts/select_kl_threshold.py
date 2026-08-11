"""Select a preregistered OPD cutoff from warm-lens validation results.

Default: the 95th percentile of first-token KL.  This tolerates the normal
late-layer mismatch on 95% of held-out examples, then treats larger divergences
at later horizons as evidence that the activation-only student should abstain.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.eval).read_text())
    first = [r["kl_by_token"][0] for r in data["detail"] if r.get("kl_by_token")]
    threshold = float(np.quantile(first, args.quantile))
    sensitivity = {}
    for q in (0.9, 0.95, 0.99):
        value = float(np.quantile(first, q))
        horizons = []
        for row in data["detail"]:
            ks = row.get("kl_by_token", [])
            stop = next((i + 1 for i, x in enumerate(ks) if x > value), len(ks))
            horizons.append(stop)
        sensitivity[str(q)] = {
            "threshold_nats": value,
            "mean_effective_horizon": float(np.mean(horizons)),
        }
    out = {
        "rule": "first-token validation KL quantile",
        "quantile": args.quantile,
        "threshold_nats": threshold,
        "n": len(first),
        "sensitivity": sensitivity,
        "source_eval": args.eval,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
