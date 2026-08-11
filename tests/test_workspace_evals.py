import json
from pathlib import Path

import pytest

from evals.judge_workspace_readouts import parse, summarise, summarise_band

OFFICIAL_COUNTS = {
    "association": 102,
    "multihop": 93,
    "multilingual": 107,
    "order-ops": 55,
    "poetry": 98,
    "typo": 96,
}


@pytest.mark.parametrize(("name", "expected"), OFFICIAL_COUNTS.items())
def test_official_workspace_dataset_is_complete(name, expected):
    path = Path("evals/datasets/official/evaluations") / f"lens-eval-{name}.json"
    data = json.loads(path.read_text())
    assert len(data["items"]) == expected
    for item in data["items"]:
        assert item["name"]
        assert item["prompt"]
        assert item["intermediates"]


def test_judge_parse_requires_one_score_per_readout():
    record = {"readouts": ["a", "b"], "intermediates": ["x"]}
    text = json.dumps({
        "readouts": [{
            "covered": [], "coherence": 2,
            "unrelated_hallucination": False, "answer_skip": False,
        }]
    })
    assert "judge_error" in parse(record, text, None)


def test_summary_distinguishes_union_from_joint_recovery():
    record = {
        "distribution": "multilingual", "mode": "raw", "layer": 42,
        "intermediates": ["big", "small"], "jlens_top": ["big"],
        "jlens_covered": ["big"],
        "scores": [
            {"covered": ["big"], "coherence": 5,
             "unrelated_hallucination": False, "answer_skip": False},
            {"covered": ["small"], "coherence": 5,
             "unrelated_hallucination": False, "answer_skip": False},
        ],
    }
    summary = summarise([record])["multilingual"]["raw"]["42"]
    assert summary["concept_recall_at_k"] == 1.0
    assert summary["joint_single_readout_recovery"] == 0.0
    assert summary["jlens_lexical_concept_recall"] == 0.5


def test_workspace_band_unions_hits_across_layers_but_not_joint_readouts():
    records = []
    for layer, concept in ((32, "big"), (42, "small")):
        records.append({
            "distribution": "multilingual", "mode": "raw",
            "name": "item", "layer": layer,
            "intermediates": ["big", "small"],
            "jlens_covered": [concept],
            "scores": [{
                "covered": [concept], "coherence": 5,
                "unrelated_hallucination": False, "answer_skip": False,
            }],
        })
    summary = summarise_band(records)["multilingual"]["raw"]
    assert summary["concept_recall_across_band"] == 1.0
    assert summary["jlens_concept_recall_across_band"] == 1.0
    assert summary["joint_single_readout_any_layer"] == 0.0
