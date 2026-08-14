"""Recompute every judged arm on ONE scale, straight from the per-record scores.

The judge emits 0/1/2. judge_fedlayer.py's *summary* divides by 2 (so
by_fed_layer is normalized 0-1) but the older breakdown_multislot.py printed the
RAW 0-2 mean, and some stored breakdown JSONs predate that fix. Reading one
number off a log and another off a breakdown is what produced a bogus "parity
with baseline" claim for several hours, so nothing here trusts a stored
aggregate: every value is recomputed from `detail`, which carries the raw
per-record judge scores.

Emits normalized 0-1 (raw in parens) per condition, plus the degenerate-output
fraction, since a low CE with collapsed text is a failure mode already seen.
"""
import argparse
import collections
import glob
import json
import os
import re

W_KEYS = ("judge_workspace", "agree_jlens", "jlens_agree", "score_jlens", "workspace", "jlens")
A_KEYS = ("judge_answer", "agree_answer", "answer_agree", "score_answer", "answer")

DEGEN = re.compile(r"^(<think>|`+\d|\d+$|)$")


def is_degenerate(text):
    """Collapsed/non-answer readout: empty, pure digits, a bare control token, or
    one short unit repeated (the '现代现代现代' failure)."""
    t = (text or "").strip()
    if len(t) < 3 or t in ("<think>", "</think>"):
        return True
    if re.fullmatch(r"[\d\s`.,:*#\-]+", t):
        return True
    # trailing repetition, not prefix-anchored: the observed collapse was
    # '6现代现代现代…', where one leading token hides an otherwise pure repeat
    for n in range(1, 9):
        if len(t) >= 4 * n and t.endswith(t[-n:] * 4):
            return True
    return False


def pick(r, keys):
    for k in keys:
        v = r.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def readouts(r):
    v = r.get("readout") or r.get("readouts") or r.get("generation")
    if isinstance(v, str):
        return [v]
    return list(v) if isinstance(v, list) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="/workspace/results/multislot_eval/*_judged.json")
    ap.add_argument("--out", default="/workspace/results/multislot_eval/ALL_rescored.json")
    args = ap.parse_args()

    table = []
    for f in sorted(glob.glob(args.glob)):
        d = json.load(open(f))
        detail = d.get("detail") if isinstance(d, dict) else (d if isinstance(d, list) else None)
        if not detail:
            print(f"  skip {os.path.basename(f)} (no per-record detail)")
            continue
        agg = collections.defaultdict(lambda: {"w": [], "a": [], "dg": []})
        for r in detail:
            w = pick(r, W_KEYS)
            if w is None:
                continue
            c = agg[r.get("condition", "?")]
            c["w"].append(w)
            a = pick(r, A_KEYS)
            if a is not None:
                c["a"].append(a)
            for t in readouts(r):
                c["dg"].append(is_degenerate(t))
        arm = os.path.basename(f).replace("_judged.json", "")
        for cond, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]["w"]) / max(1, len(kv[1]["w"]))):
            n = len(v["w"])
            raw = sum(v["w"]) / max(1, n)
            var = sum((x - raw) ** 2 for x in v["w"]) / max(1, n - 1) if n > 1 else 0.0
            table.append({
                "arm": arm, "condition": cond, "n": n,
                "workspace": raw / 2, "workspace_raw02": raw,
                "se": (var ** 0.5) / max(1, n ** 0.5) / 2,
                "answer": (sum(v["a"]) / len(v["a"]) / 2) if v["a"] else None,
                "degenerate_frac": (sum(v["dg"]) / len(v["dg"])) if v["dg"] else None,
            })

    table.sort(key=lambda r: -r["workspace"])
    print(f"\n{'arm':26s} {'condition':22s} {'n':>5s}  workspace(0-1)   raw   degen")
    for r in table:
        dg = f"{r['degenerate_frac']:5.0%}" if r["degenerate_frac"] is not None else "    -"
        print(f"{r['arm'][:26]:26s} {r['condition'][:22]:22s} {r['n']:5d}  "
              f"{r['workspace']:.3f}±{r['se']:.3f}   ({r['workspace_raw02']:.3f}) {dg}")
    json.dump(table, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out} ({len(table)} rows) — all values NORMALIZED 0-1")


if __name__ == "__main__":
    main()
