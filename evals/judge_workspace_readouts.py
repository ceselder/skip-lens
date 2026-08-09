"""Judge coherent multi-concept recovery on official workspace prompts.

Unlike token pass@k, this scores whether a *single natural-language readout*
coherently expresses several expected intermediates together.  Same-distribution
shuffled activations estimate the judge/readout false-positive floor.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from ._batch_llm import batch_call
    from ._llm import call as llm_call
except ImportError:  # direct `python evals/judge_workspace_readouts.py`
    from _batch_llm import batch_call
    from _llm import call as llm_call

MODEL = "claude-sonnet-5"
JUDGE = """You are evaluating a natural-language lens that reads a language model's internal workspace.

The lens was applied while the model processed this prompt:
<prompt>
{prompt}
</prompt>

Expected silent intermediate concepts:
{concepts}

The surface answer/target, when defined, is: {target}

Independent sampled readouts of the SAME activation:
{readouts}

For each readout separately:
1. Mark which expected intermediate concepts it states or clearly implies. Do not infer a concept from the original prompt; it must be present in the readout.
2. Score coherence from 1 (broken word salad) to 5 (clear, internally coherent phrase or continuation).
3. Mark unrelated_hallucination true if it introduces specific content unrelated to both the prompt and expected intermediates.
4. Mark answer_skip true if it merely jumps to the surface target while expressing none of the expected intermediates.

Return JSON only, preserving readout order:
{{"readouts": [{{"covered": ["exact expected strings only"], "coherence": 1, "unrelated_hallucination": false, "answer_skip": false}}]}}"""


def make_prompt(record):
    concepts = record["intermediates"]
    readouts = record["readouts"]
    return JUDGE.format(
        prompt=record["prompt"][:1800],
        concepts="\n".join(f"- {x}" for x in concepts),
        target=record.get("target") or "(none)",
        readouts="\n".join(f"[{i}] {x}" for i, x in enumerate(readouts)),
    )


def parse(record, text, err):
    if text is None:
        return {**record, "judge_error": err}
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
        scores = obj["readouts"]
        expected_count = len(record["readouts"])
        if len(scores) != expected_count:
            raise ValueError(
                f"returned {len(scores)} scores for {expected_count} readouts"
            )
        return {**record, "scores": scores}
    except (ValueError, KeyError, TypeError) as exc:
        return {**record, "judge_error": f"{exc}: {text[:240]}"}


def lexical_jlens_coverage(record):
    if "jlens_covered" in record:
        return record["jlens_covered"]
    text = " ".join(record.get("jlens_top", [])).lower()
    return [c for c in record["intermediates"]
            if re.search(r"(?<!\w)" + re.escape(c.lower()) + r"(?!\w)", text)]


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarise(records):
    buckets = {}
    for record in records:
        if "scores" not in record:
            continue
        key = (record["distribution"], record["mode"], record["layer"])
        bucket = buckets.setdefault(key, {
            "concept_recall": [], "joint_recovery": [], "coherence": [],
            "hallucination": [], "answer_skip": [], "jlens_concept_recall": [],
        })
        expected = {x.lower() for x in record["intermediates"]}
        per_readout_sets = []
        for score in record["scores"]:
            covered = {str(x).lower() for x in score.get("covered", [])} & expected
            per_readout_sets.append(covered)
            bucket["coherence"].append(float(score.get("coherence", 1)))
            bucket["hallucination"].append(bool(score.get("unrelated_hallucination")))
            bucket["answer_skip"].append(bool(score.get("answer_skip")))
        union = set().union(*per_readout_sets) if per_readout_sets else set()
        bucket["concept_recall"].append(len(union) / max(1, len(expected)))
        bucket["joint_recovery"].append(any(s == expected for s in per_readout_sets))
        bucket["jlens_concept_recall"].append(
            len(lexical_jlens_coverage(record)) / max(1, len(expected)))

    out = {}
    for (dist, mode, layer), values in buckets.items():
        out.setdefault(dist, {}).setdefault(mode, {})[str(layer)] = {
            "n_items": len(values["concept_recall"]),
            "concept_recall_at_k": mean(values["concept_recall"]),
            "joint_single_readout_recovery": mean(values["joint_recovery"]),
            "mean_coherence_1_5": mean(values["coherence"]),
            "unrelated_hallucination_rate": mean(values["hallucination"]),
            "answer_skip_rate": mean(values["answer_skip"]),
            "jlens_lexical_concept_recall": mean(values["jlens_concept_recall"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sync", action="store_true",
                    help="use synchronous low-priority calls (small smoke tests only)")
    args = ap.parse_args()
    data = json.loads(Path(args.input).read_text())
    source_records = data["records"]
    if args.sync:
        def one(record):
            return parse(record, *llm_call(
                make_prompt(record), MODEL, max_tokens=1024, temperature=0.0))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(one, source_records))
    else:
        results = batch_call([make_prompt(record) for record in source_records],
                             MODEL, 1024, args.out + ".batch.json")
        records = [parse(record, text, err)
                   for record, (text, err) in zip(source_records, results)]
    result = {
        "meta": {**data["meta"], "judge_model": MODEL},
        "summary": summarise(records),
        "judge_errors": sum("judge_error" in x for x in records),
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}; judge_errors={result['judge_errors']}")


if __name__ == "__main__":
    main()
