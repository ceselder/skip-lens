import json
import sys

import numpy as np
import pytest

from evals import analyze_scaled_opd


def _evaluation(checkpoint, nll):
    detail = []
    for index, offset in enumerate((0.0, 0.1)):
        detail.append({
            "index": index,
            "reference_student_nll": nll + offset,
            "reference_mean_kl": nll + offset + 0.2,
            "reference_mean_reverse_kl": nll + offset + 0.1,
            "reference_top1_agreement": 0.5 - offset,
            "exact_prefix_tokens": 1 + index,
            "length": 8,
            "ended_eos": False,
        })
    aggregate = {
        "reference_student_nll": nll,
        "reference_mean_kl": nll + 0.2,
        "reference_mean_reverse_kl": nll + 0.1,
        "reference_top1_agreement": 0.5,
        "mean_length": 8.0,
    }
    return {
        "checkpoint": checkpoint,
        "aggregate": aggregate,
        "by_horizon": {},
        "detail": detail,
    }


def test_selects_best_opd_and_compares_same_step(tmp_path, monkeypatch):
    results = tmp_path / "results"
    checkpoints = tmp_path / "checkpoints"
    results.mkdir()
    for arm in ("opd", "sft_matched"):
        root = checkpoints / arm
        root.mkdir(parents=True)
        (root / "run_summary.json").write_text(json.dumps({
            "optimized_tokens": 1024, "wall_seconds": 10.0,
        }))
        (root / "metrics.jsonl").write_text(json.dumps({
            "step_seconds": 1.0, "loss": 1.0,
        }) + "\n")
        for step in (50, 100):
            for feed in ("L62", "L42"):
                # OPD is best at 100. SFT is independently best at 50, ensuring
                # the primary control still comes from OPD's selected update.
                if arm == "opd":
                    nll = 2.0 if step == 50 else 1.5
                else:
                    nll = 1.0 if step == 50 else 1.2
                payload = _evaluation(f"{arm}/{step}", nll)
                (results / f"{arm}_{step:07d}_{feed}.json").write_text(
                    json.dumps(payload)
                )

    out = tmp_path / "analysis.json"
    monkeypatch.setattr(sys, "argv", [
        "analyze_scaled_opd", "--results", str(results),
        "--checkpoints", str(checkpoints), "--out", str(out),
        "--bootstrap-samples", "100",
    ])
    analyze_scaled_opd.main()
    data = json.loads(out.read_text())
    assert data["selection"]["selected_opd_step"] == 100
    assert data["selection"]["best_sft_nll_step"] == 50
    assert data["selection"]["matched_sft_checkpoint"].endswith("iter_0000100")
    assert (
        data["paired_comparisons"]["L62"]["reference_student_nll"]
        ["opd_minus_sft"]
        == pytest.approx(0.3)
    )


def test_paired_quality_uses_matching_rows():
    rows = []
    for arm, coherence in (("opd", 4), ("sft_matched", 3)):
        for index in range(3):
            rows.append({
                "arm": arm, "feed": "L62", "index": index,
                "quality": {
                    "coherence": coherence, "support": coherence,
                    "hallucination": arm == "sft_matched",
                    "premature_eos": False,
                },
            })
    result = analyze_scaled_opd.paired_quality(
        rows, "L62", 100, np.random.default_rng(0),
    )
    assert result["coherence"]["opd_minus_sft"] == 1
    assert result["hallucination"]["opd_minus_sft"] == -1
