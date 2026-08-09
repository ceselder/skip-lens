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

There are exactly {readout_count} readouts in the JSON array above. Return
exactly {readout_count} score objects in the same order; embedded newlines are
part of one string, not additional readouts.

For each readout separately:
1. Mark which expected intermediate concepts it states or clearly implies. Do not infer a concept from the original prompt; it must be present in the readout.
2. Score coherence from 1 (broken word salad) to 5 (clear, internally coherent phrase or continuation).
3. Mark unrelated_hallucination true if it introduces specific content unrelated to both the prompt and expected intermediates.
4. Mark answer_skip true if it merely jumps to the surface target while expressing none of the expected intermediates.

Return JSON only, preserving readout order:
{{"readouts": [{{"covered": ["exact expected strings only"], "coherence": 1, "unrelated_hallucination": false, "answer_skip": false}}]}}"""


def record_key(record):
    return (
        record["distribution"], record["name"], int(record["layer"]), record["mode"]
    )


def make_prompt(record):
    concepts = record["intermediates"]
    readouts = record["readouts"]
    return JUDGE.format(
        prompt=record["prompt"][:1800],
        concepts="\n".join(f"- {x}" for x in concepts),
        target=record.get("target") or "(none)",
        readouts=json.dumps(readouts, ensure_ascii=False),
        readout_count=len(readouts),
    )


def parse(record, text, err):
    clean_record = {k: v for k, v in record.items()
                    if k not in {"scores", "judge_error"}}
    if text is None:
        return {**clean_record, "judge_error": err}
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
        scores = obj["readouts"]
        expected_count = len(record["readouts"])
        if len(scores) != expected_count:
            raise ValueError(
                f"returned {len(scores)} scores for {expected_count} readouts"
            )
        return {**clean_record, "scores": scores}
    except (ValueError, KeyError, TypeError) as exc:
        return {**clean_record, "judge_error": f"{exc}: {text[:240]}"}


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


def summarise_band(records):
    """Collapse layers using the paper's any-hit-in-workspace-band rule."""
    items = {}
    for record in records:
        if "scores" not in record:
            continue
        key = (record["distribution"], record["mode"], record["name"])
        item = items.setdefault(key, {
            "expected": {x.lower() for x in record["intermediates"]},
            "covered": set(), "jlens_covered": set(), "joint": False,
        })
        for score in record["scores"]:
            covered = ({str(x).lower() for x in score.get("covered", [])}
                       & item["expected"])
            item["covered"].update(covered)
            item["joint"] |= covered == item["expected"]
        item["jlens_covered"].update(
            x.lower() for x in lexical_jlens_coverage(record)
        )

    buckets = {}
    for (distribution, mode, _name), item in items.items():
        key = (distribution, mode)
        bucket = buckets.setdefault(key, {
            "concept_recall": [], "joint": [], "jlens_recall": [],
        })
        denom = max(1, len(item["expected"]))
        bucket["concept_recall"].append(len(item["covered"]) / denom)
        bucket["joint"].append(item["joint"])
        bucket["jlens_recall"].append(len(item["jlens_covered"] & item["expected"]) / denom)

    out = {}
    for (distribution, mode), values in buckets.items():
        out.setdefault(distribution, {})[mode] = {
            "n_items": len(values["concept_recall"]),
            "concept_recall_across_band": mean(values["concept_recall"]),
            "joint_single_readout_any_layer": mean(values["joint"]),
            "jlens_concept_recall_across_band": mean(values["jlens_recall"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="judge output budget; must include Sonnet 5 reasoning tokens")
    ap.add_argument("--layers", default=None,
                    help="optional comma-separated layer subset")
    ap.add_argument("--modes", default=None,
                    help="optional comma-separated readout-mode subset")
    ap.add_argument("--retry-errors-only", action="store_true",
                    help="input is a judged file; repair only its errored records")
    ap.add_argument("--sync", action="store_true",
                    help="use synchronous low-priority calls (small smoke tests only)")
    args = ap.parse_args()
    data = json.loads(Path(args.input).read_text())
    all_records = data["records"]
    source_records = all_records
    if args.retry_errors_only:
        source_records = [x for x in source_records if "judge_error" in x]
    if args.layers:
        selected_layers = {int(x) for x in args.layers.split(",")}
        source_records = [x for x in source_records if x["layer"] in selected_layers]
    if args.modes:
        selected_modes = set(args.modes.split(","))
        source_records = [x for x in source_records if x["mode"] in selected_modes]
    if args.sync:
        def one(record):
            return parse(record, *llm_call(
                make_prompt(record), MODEL, max_tokens=args.max_tokens, temperature=0.0))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            records = list(pool.map(one, source_records))
    else:
        prompts = [make_prompt(record) for record in source_records]
        results = batch_call(
            prompts, MODEL, args.max_tokens, args.out + ".batch.json")
        records = [parse(record, text, err)
                   for record, (text, err) in zip(source_records, results)]
        retry_indices = [i for i, record in enumerate(records)
                         if "judge_error" in record]
        if retry_indices:
            print(f"[judge] retrying {len(retry_indices)} malformed/failed responses", flush=True)
            retry_results = batch_call(
                [prompts[i] for i in retry_indices], MODEL, args.max_tokens * 2,
                args.out + ".retry.batch.json",
            )
            for i, (text, err) in zip(retry_indices, retry_results):
                records[i] = parse(source_records[i], text, err)
    if args.retry_errors_only:
        repaired = {record_key(record): record for record in records}
        records = [repaired.get(record_key(record), record) for record in all_records]

    result = {
        "meta": {
            **data["meta"], "judge_model": MODEL,
            "judge_layers": sorted({x["layer"] for x in source_records}),
            "judge_modes": sorted({x["mode"] for x in source_records}),
        },
        "summary": summarise(records),
        "summary_band": summarise_band(records),
        "judge_errors": sum("judge_error" in x for x in records),
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}; judge_errors={result['judge_errors']}")


if __name__ == "__main__":
    main()
