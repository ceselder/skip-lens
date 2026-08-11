"""Summarize the fixed-eight-token OPD experiment with paired uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LOWER_IS_BETTER = (
    "reference_mean_kl",
    "reference_student_nll",
)
HIGHER_IS_BETTER = (
    "reference_top1_agreement",
    "exact_prefix_tokens",
)


def paired_bootstrap(
    opd_rows: list[dict],
    control_rows: list[dict],
    metric: str,
    samples: int,
    seed: int,
) -> dict:
    opd = {int(row["index"]): float(row[metric]) for row in opd_rows}
    control = {int(row["index"]): float(row[metric]) for row in control_rows}
    indices = sorted(set(opd) & set(control))
    if not indices:
        raise ValueError(f"no paired rows for {metric}")
    delta = np.asarray([opd[i] - control[i] for i in indices], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(samples, len(delta)), replace=True).mean(axis=1)
    lower_is_better = metric in LOWER_IS_BETTER
    return {
        "n": len(delta),
        "opd_minus_control": float(delta.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "opd_better_probability": float(
            np.mean(draws < 0) if lower_is_better else np.mean(draws > 0)
        ),
        "opd_std": float(np.std([opd[i] for i in indices], ddof=1)),
        "control_std": float(np.std([control[i] for i in indices], ddof=1)),
    }


def load_eval(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = Path(args.results)
    checkpoints = Path(args.checkpoints)
    arms = ("warm", "opd8", "sft8_tokens", "sft8_time")
    feeds = ("L62", "L42")
    datasets = ("normal", "repeat")
    evaluations: dict[str, dict] = {}
    comparisons: dict[str, dict] = {}

    for dataset in datasets:
        evaluations[dataset] = {}
        comparisons[dataset] = {}
        for feed in feeds:
            condition = {
                arm: load_eval(results / f"{dataset}_{arm}_{feed}_eval.json")
                for arm in arms
            }
            evaluations[dataset][feed] = {
                arm: value["aggregate"] for arm, value in condition.items()
            }
            comparisons[dataset][feed] = {}
            for control in ("warm", "sft8_tokens", "sft8_time"):
                comparisons[dataset][feed][control] = {
                    metric: paired_bootstrap(
                        condition["opd8"]["detail"],
                        condition[control]["detail"],
                        metric,
                        args.bootstrap_samples,
                        args.seed,
                    )
                    for metric in (*LOWER_IS_BETTER, *HIGHER_IS_BETTER)
                }

    training: dict[str, dict] = {}
    for dataset in datasets:
        training[dataset] = {}
        for arm in ("opd8", "sft8_tokens", "sft8_time"):
            summary = json.loads(
                (checkpoints / f"{dataset}_{arm}" / "run_summary.json").read_text()
            )
            summary["optimized_tokens_per_second"] = (
                summary["optimized_tokens"] / summary["wall_seconds"]
            )
            training[dataset][arm] = summary

    output = {
        "bootstrap_samples": args.bootstrap_samples,
        "training": training,
        "evaluations": evaluations,
        "paired_comparisons": comparisons,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
