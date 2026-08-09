"""Paired comparison of two judged natural-language workspace readers."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def record_key(record: dict) -> tuple[str, str, int, str]:
    return (
        record["distribution"], record["name"], int(record["layer"]), record["mode"]
    )


def metrics(record: dict) -> dict[str, float]:
    expected = {str(x).lower() for x in record["intermediates"]}
    covered_sets = [
        {str(x).lower() for x in score.get("covered", [])} & expected
        for score in record["scores"]
    ]
    union = set().union(*covered_sets) if covered_sets else set()
    return {
        "workspace_recall": len(union) / max(1, len(expected)),
        "joint_recovery": float(any(x == expected for x in covered_sets)),
        "coherence": float(np.mean([
            float(x.get("coherence", 1)) for x in record["scores"]
        ])),
        "hallucination": float(np.mean([
            bool(x.get("unrelated_hallucination")) for x in record["scores"]
        ])),
        "answer_skip": float(np.mean([
            bool(x.get("answer_skip")) for x in record["scores"]
        ])),
    }


def bootstrap_delta(
    normal: np.ndarray,
    repeat: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    delta = repeat - normal
    n = len(delta)
    draws = delta[rng.integers(0, n, size=(samples, n))].mean(axis=1)
    return {
        "normal": float(normal.mean()),
        "repeat": float(repeat.mean()),
        "delta_repeat_minus_normal": float(delta.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "n": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", required=True)
    ap.add_argument("--repeat", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap-samples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples-per-distribution", type=int, default=2)
    args = ap.parse_args()

    normal_data = json.loads(Path(args.normal).read_text())
    repeat_data = json.loads(Path(args.repeat).read_text())
    normal_records = {
        record_key(x): x for x in normal_data["records"] if "scores" in x
    }
    repeat_records = {
        record_key(x): x for x in repeat_data["records"] if "scores" in x
    }
    keys = sorted(normal_records.keys() & repeat_records.keys())
    if not keys:
        raise ValueError("no complete paired workspace records")
    missing = {
        "normal_only": len(normal_records.keys() - repeat_records.keys()),
        "repeat_only": len(repeat_records.keys() - normal_records.keys()),
        "normal_judge_errors": normal_data.get("judge_errors", 0),
        "repeat_judge_errors": repeat_data.get("judge_errors", 0),
    }

    rows = []
    for key in keys:
        nrec, rrec = normal_records[key], repeat_records[key]
        rows.append({
            "key": key,
            "normal": metrics(nrec), "repeat": metrics(rrec),
            "normal_record": nrec, "repeat_record": rrec,
        })

    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        distribution, _name, layer, _mode = row["key"]
        groups[(distribution, layer)].append(row)
        groups[("all", layer)].append(row)
    rng = np.random.default_rng(args.seed)
    summary = defaultdict(dict)
    metric_names = list(rows[0]["normal"])
    for (distribution, layer), values in sorted(groups.items()):
        summary[distribution][str(layer)] = {
            metric: bootstrap_delta(
                np.asarray([x["normal"][metric] for x in values]),
                np.asarray([x["repeat"][metric] for x in values]),
                args.bootstrap_samples, rng,
            )
            for metric in metric_names
        }

    examples = []
    for distribution in sorted({x["key"][0] for x in rows}):
        candidates = [x for x in rows
                      if x["key"][0] == distribution and x["key"][2] == 42]
        candidates.sort(key=lambda x: (
            x["repeat"]["workspace_recall"] - x["normal"]["workspace_recall"],
            x["repeat"]["coherence"] - x["normal"]["coherence"],
            x["normal"]["hallucination"] - x["repeat"]["hallucination"],
        ), reverse=True)
        for row in candidates[:args.examples_per_distribution]:
            nrec, rrec = row["normal_record"], row["repeat_record"]
            examples.append({
                "distribution": distribution, "name": row["key"][1], "layer": 42,
                "prompt": nrec["prompt"], "intermediates": nrec["intermediates"],
                "target": nrec.get("target"),
                "normal_readouts": nrec["readouts"], "normal_scores": nrec["scores"],
                "repeat_readouts": rrec["readouts"], "repeat_scores": rrec["scores"],
            })

    result = {
        "meta": {
            "normal": args.normal, "repeat": args.repeat,
            "bootstrap_samples": args.bootstrap_samples, "seed": args.seed,
            "paired_records": len(keys), **missing,
        },
        "summary": dict(summary),
        "examples": examples,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result["meta"], indent=2))


if __name__ == "__main__":
    main()
