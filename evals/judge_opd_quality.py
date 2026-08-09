"""Sonnet-5 coherence and unsupported-specificity judge for OPD readouts."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:
    from ._batch_llm import batch_call
    from ._llm import call as llm_call
except ImportError:  # direct `python evals/judge_opd_quality.py`
    from _batch_llm import batch_call
    from _llm import call as llm_call

MODEL = "claude-sonnet-5"
PROMPT = """You are evaluating an activation-only forecast of what a language model will say next.

SOURCE CONTEXT (tail):
{context}

ONE REFERENCE CONTINUATION:
{reference}

ACTIVATION-ONLY READOUT:
{readout}

Score the readout itself. The reference is one possible continuation, not the only acceptable one.
- coherence: 1 (broken/incoherent) to 5 (fully fluent and internally coherent)
- support: 1 (mostly arbitrary specificity or contradiction) to 5 (well supported as a plausible continuation of the source)
- hallucination: true only when the readout invents concrete unsupported or contradictory details; a different but plausible continuation is not a hallucination.
- premature_eos: true if the readout is empty or stops before expressing any meaningful continuation.

Return JSON only:
{{"coherence": 1, "support": 1, "hallucination": false, "premature_eos": false, "reason": "brief"}}"""


def make_prompt(row):
    return PROMPT.format(context=row.get("context", "")[-1200:],
                         reference=row.get("reference", "")[:800],
                         readout=row.get("readout", "")[:800])


def parse(row, text, err):
    if text is None:
        return {**row, "judge_error": err}
    try:
        start = text.index("{")
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
        return {**row, "quality": obj}
    except (ValueError, KeyError, TypeError) as exc:
        return {**row, "judge_error": f"parse: {exc}: {text[:200]}"}


def mean(values):
    return sum(values) / len(values) if values else None


def summarise_quality(rows):
    return {
        "n": len(rows),
        "coherence": mean([float(x["coherence"]) for x in rows]),
        "support": mean([float(x["support"]) for x in rows]),
        "hallucination_rate": mean([bool(x["hallucination"]) for x in rows]),
        "premature_eos_rate": mean([bool(x["premature_eos"]) for x in rows]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--sync", action="store_true",
                    help="use synchronous low-priority calls (small smoke tests only)")
    args = ap.parse_args()
    source = json.loads(Path(args.input).read_text())
    rows = source["detail"]
    if args.sync:
        def one(row):
            return parse(row, *llm_call(make_prompt(row), MODEL, max_tokens=512, temperature=0.0))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            detail = list(pool.map(one, rows))
    else:
        results = batch_call([make_prompt(row) for row in rows], MODEL, 512,
                             args.out + ".batch.json")
        detail = [parse(row, text, err) for row, (text, err) in zip(rows, results)]
    valid = [x["quality"] for x in detail if "quality" in x]
    summary = {**summarise_quality(valid), "errors": len(detail) - len(valid)}
    grouped = {}
    for row in detail:
        if "quality" not in row:
            continue
        key = (row.get("dataset", "unknown"), row.get("arm", "unknown"),
               row.get("feed", row.get("feed_col", "unknown")))
        grouped.setdefault(key, []).append(row["quality"])
    by_condition = {}
    for (dataset, arm, feed), values in grouped.items():
        by_condition.setdefault(dataset, {}).setdefault(arm, {})[feed] = (
            summarise_quality(values)
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "model": MODEL,
        "summary": summary,
        "by_condition": by_condition,
        "detail": detail,
    }, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
