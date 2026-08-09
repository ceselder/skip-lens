"""Summarize corrected reverse-KL OPD against token-matched SFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = {
    "reference_student_nll": False,
    "reference_mean_kl": False,
    "reference_mean_reverse_kl": False,
    "reference_top1_agreement": True,
    "exact_prefix_tokens": True,
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def paired_delta(
    opd_rows: list[dict],
    sft_rows: list[dict],
    metric: str,
    higher_is_better: bool,
    samples: int,
    rng: np.random.Generator,
) -> dict:
    opd = {int(x["index"]): float(x[metric]) for x in opd_rows}
    sft = {int(x["index"]): float(x[metric]) for x in sft_rows}
    indices = sorted(opd.keys() & sft.keys())
    delta = np.asarray([opd[i] - sft[i] for i in indices])
    draws = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(1)
    return {
        "n": len(delta),
        "opd_minus_sft": float(delta.mean()),
        "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
        "opd_better_probability": float(
            np.mean(draws > 0) if higher_is_better else np.mean(draws < 0)
        ),
    }


def quality_delta(
    rows: list[dict],
    dataset: str,
    feed: str,
    samples: int,
    rng: np.random.Generator,
) -> dict:
    def arm(name: str) -> dict[int, dict]:
        return {
            int(x["index"]): x["quality"] for x in rows
            if x.get("dataset") == dataset and x.get("arm") == name
            and x.get("feed") == feed and "quality" in x
        }

    opd, sft = arm("opd"), arm("sft")
    indices = sorted(opd.keys() & sft.keys())
    result = {}
    for metric, higher_is_better in (
        ("coherence", True), ("support", True),
        ("hallucination", False), ("premature_eos", False),
    ):
        delta = np.asarray([
            float(opd[i][metric]) - float(sft[i][metric]) for i in indices
        ])
        draws = delta[rng.integers(0, len(delta), size=(samples, len(delta)))].mean(1)
        result[metric] = {
            "n": len(delta), "opd_minus_sft": float(delta.mean()),
            "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
            "opd_better_probability": float(
                np.mean(draws > 0) if higher_is_better else np.mean(draws < 0)
            ),
        }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quality", default=None)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    results, checkpoints = Path(args.results), Path(args.checkpoints)
    rng = np.random.default_rng(args.seed)

    evaluations = {}
    comparisons = {}
    for dataset in ("normal", "repeat"):
        evaluations[dataset], comparisons[dataset] = {}, {}
        for feed in ("L62", "L42"):
            opd = load(results / f"{dataset}_opd_{feed}_eval.json")
            sft = load(results / f"{dataset}_sft_{feed}_eval.json")
            evaluations[dataset][feed] = {
                "opd": opd["aggregate"], "sft": sft["aggregate"],
            }
            comparisons[dataset][feed] = {
                metric: paired_delta(
                    opd["detail"], sft["detail"], metric, higher,
                    args.bootstrap_samples, rng,
                )
                for metric, higher in METRICS.items()
            }

    checkpoint_dirs = {
        "normal": {
            "opd": "normal_true_opd8", "sft": "normal_sft8_tokens",
        },
        "repeat": {
            "opd": "repeat_true_opd8_strongwarm", "sft": "repeat_sft8_strongwarm",
        },
    }
    training = {}
    for dataset, arms in checkpoint_dirs.items():
        training[dataset] = {}
        for arm, directory in arms.items():
            summary = load(checkpoints / directory / "run_summary.json")
            summary["optimized_tokens_per_second"] = (
                summary["optimized_tokens"] / summary["wall_seconds"]
            )
            training[dataset][arm] = summary

    output = {
        "bootstrap_samples": args.bootstrap_samples,
        "training": training, "evaluations": evaluations,
        "paired_comparisons": comparisons,
    }
    if args.quality:
        quality = load(Path(args.quality))
        output["quality"] = {
            "summary": quality["summary"],
            "by_condition": quality["by_condition"],
            "paired_comparisons": {
                dataset: {
                    feed: quality_delta(
                        quality["detail"], dataset, feed,
                        args.bootstrap_samples, rng,
                    )
                    for feed in ("L62", "L42")
                }
                for dataset in ("normal", "repeat")
            },
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
