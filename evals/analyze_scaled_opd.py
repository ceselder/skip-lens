"""Analyze the checkpoint trajectory for scaled FineWeb OPD versus matched SFT.

Checkpoint selection is deliberately explicit: the upload candidate is the OPD
checkpoint with the lowest teacher-forced L62 validation NLL.  The comparison
control is the SFT checkpoint at the *same optimizer update*, preserving the
token match.  Independently best checkpoints are also recorded, but are not
used as the primary paired comparison.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


EVAL_RE = re.compile(r"^(opd|sft_matched)_(\d+)_L(42|62)\.json$")
PAIRED_METRICS = {
    "reference_student_nll": False,
    "reference_mean_kl": False,
    "reference_mean_reverse_kl": False,
    "reference_top1_agreement": True,
    "exact_prefix_tokens": True,
    "length": True,
    "ended_eos": False,
}


def load_evaluations(results: Path) -> dict[str, dict[str, dict[int, dict]]]:
    evaluations: dict[str, dict[str, dict[int, dict]]] = {
        arm: {feed: {} for feed in ("L62", "L42")}
        for arm in ("opd", "sft_matched")
    }
    for path in results.glob("*.json"):
        match = EVAL_RE.match(path.name)
        if not match:
            continue
        arm, step_text, layer = match.groups()
        evaluations[arm][f"L{layer}"][int(step_text)] = json.loads(path.read_text())
    for arm, feeds in evaluations.items():
        for feed, by_step in feeds.items():
            if not by_step:
                raise FileNotFoundError(f"no {arm} {feed} evaluations in {results}")
    return evaluations


def paired_delta(
    left_rows: list[dict],
    right_rows: list[dict],
    metric: str,
    higher_is_better: bool,
    samples: int,
    rng: np.random.Generator,
) -> dict:
    left = {int(row["index"]): float(row[metric]) for row in left_rows}
    right = {int(row["index"]): float(row[metric]) for row in right_rows}
    if left.keys() != right.keys():
        raise ValueError(f"paired rows differ for {metric}")
    indices = sorted(left)
    delta = np.asarray([left[index] - right[index] for index in indices])
    draw_indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    draws = delta[draw_indices].mean(axis=1)
    return {
        "n": len(delta),
        "opd_minus_sft": float(delta.mean()),
        "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
        "opd_better_probability": float(
            np.mean(draws > 0) if higher_is_better else np.mean(draws < 0)
        ),
    }


def compact_eval(data: dict) -> dict:
    return {
        "checkpoint": data["checkpoint"],
        "aggregate": data["aggregate"],
        "by_horizon": data["by_horizon"],
    }


def load_training(checkpoints: Path) -> dict:
    output = {}
    for arm in ("opd", "sft_matched"):
        root = checkpoints / arm
        summary = json.loads((root / "run_summary.json").read_text())
        summary["optimized_tokens_per_second"] = (
            summary["optimized_tokens"] / summary["wall_seconds"]
        )
        microsteps = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        output[arm] = {
            "summary": summary,
            "mean_microstep_seconds": float(np.mean([x["step_seconds"] for x in microsteps])),
            "final_microstep": microsteps[-1],
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    evaluations = load_evaluations(Path(args.results))
    common_steps = sorted(
        set(evaluations["opd"]["L62"]) & set(evaluations["sft_matched"]["L62"])
        & set(evaluations["opd"]["L42"]) & set(evaluations["sft_matched"]["L42"])
    )
    if not common_steps:
        raise ValueError("no optimizer updates have all four matched evaluations")

    best_opd_nll_step = min(
        common_steps,
        key=lambda step: (
            evaluations["opd"]["L62"][step]["aggregate"]["reference_student_nll"],
            step,
        ),
    )
    best_opd_reverse_kl_step = min(
        common_steps,
        key=lambda step: (
            evaluations["opd"]["L62"][step]["aggregate"]["reference_mean_reverse_kl"],
            step,
        ),
    )
    best_sft_nll_step = min(
        common_steps,
        key=lambda step: (
            evaluations["sft_matched"]["L62"][step]["aggregate"]["reference_student_nll"],
            step,
        ),
    )

    rng = np.random.default_rng(args.seed)
    paired = {}
    selected = {}
    for feed in ("L62", "L42"):
        opd = evaluations["opd"][feed][best_opd_nll_step]
        sft = evaluations["sft_matched"][feed][best_opd_nll_step]
        selected[feed] = {
            "opd": compact_eval(opd),
            "sft_matched": compact_eval(sft),
        }
        paired[feed] = {
            metric: paired_delta(
                opd["detail"], sft["detail"], metric, higher,
                args.bootstrap_samples, rng,
            )
            for metric, higher in PAIRED_METRICS.items()
        }

    trajectory = {
        feed: {
            arm: {
                str(step): compact_eval(evaluations[arm][feed][step])
                for step in common_steps
            }
            for arm in ("opd", "sft_matched")
        }
        for feed in ("L62", "L42")
    }
    checkpoints = Path(args.checkpoints)
    output = {
        "selection": {
            "criterion": "minimum OPD L62 teacher-forced reference_student_nll",
            "selected_opd_step": best_opd_nll_step,
            "selected_opd_checkpoint": str(
                checkpoints / "opd" / f"iter_{best_opd_nll_step:07d}"
            ),
            "matched_sft_checkpoint": str(
                checkpoints / "sft_matched" / f"iter_{best_opd_nll_step:07d}"
            ),
            "best_opd_reverse_kl_step": best_opd_reverse_kl_step,
            "best_sft_nll_step": best_sft_nll_step,
            "caveat": (
                "Checkpoint selection and reported validation metrics use the same "
                "held-out split; trajectory estimates are descriptive, not an "
                "unbiased post-selection test."
            ),
        },
        "common_steps": common_steps,
        "selected_evaluations": selected,
        "paired_comparisons": paired,
        "trajectory": trajectory,
        "training": load_training(checkpoints),
        "bootstrap_samples": args.bootstrap_samples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(json.dumps({
        "selection": output["selection"],
        "L62": selected["L62"],
        "L62_paired": paired["L62"],
        "training": output["training"],
    }, indent=2))


if __name__ == "__main__":
    main()
