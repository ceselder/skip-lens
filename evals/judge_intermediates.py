"""Score a read-out on the A.6 intermediate-concept probe, with Claude as the judge.

The paper's lens emits ranked TOKENS, so recovery is "the intermediate appears among the top-k
tokens of the lens readout at any layer". Our read-out emits natural-language phrases, so token
membership does not apply and a judge decides presence instead. pass@k becomes pass@k-bullets:
recovery when only the first k bullets are shown, reported at several k plus the paper's summary
statistic (AUC of pass@k against log k, normalised so that always-recovered-at-k=1 scores 1).

THE CONTROL IS THE WHOLE EVAL. An agreeable judge shown eight vague phrases will happily assent to
almost any concept, and a bare hit-rate cannot distinguish a working read-out from a compliant
grader. So every item is judged on its TRUE intermediate and on DECOY intermediates borrowed from
other items in the same distribution, interleaved and indistinguishable to the judge. The headline
number is discrimination = hit rate minus decoy false-positive rate. If those two are equal, the
read-out carries nothing and no amount of hit rate rescues it.

Two further guards against a judge that scores well for the wrong reason:
  - The judge never sees the prompt. Given "the color of the planet fourth from the Sun is" and
    the candidate "Mars", any competent model answers yes from world knowledge alone, regardless
    of what the bullets say. It sees only the bullets and one candidate concept.
  - Candidate order is shuffled per item, so a positional habit cannot mimic discrimination.

Read-outs file format (produced on the GPU box; this script needs no GPU):
    {"<distribution>": {"<item index>": ["bullet 1", "bullet 2", ...]}, ...}

    python evals/judge_intermediates.py --readouts run/evals/readouts_dictmp.json --tag dict_mp
    python evals/judge_intermediates.py --selftest      # validates the judge, needs no read-outs
"""
import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from _llm import call as llm_call

MODEL = "claude-sonnet-5"          # latest sonnet
HERE = Path(__file__).parent
DISTS = ["multihop", "multilingual", "order_of_ops", "poetry", "typo", "association"]

JUDGE = """You are grading whether a specific concept is present in a list of phrases that were \
decoded from a language model's internal activation.

The phrases:
{bullets}

The concept to look for: "{concept}"

Is this concept present in, or clearly implied by, the phrases above? Judge only the phrases. Do \
not reason about what a model might plausibly have been thinking, and do not give credit for \
generic topical proximity. A concept counts as present if a phrase names it, names an unambiguous \
synonym or inflection of it, or states something that could only be about it.

Answer with a JSON object and nothing else:
{{"present": true or false, "evidence": "the phrase that shows it, or empty string"}}"""


def judge_one(bullets, concept):
    msg = JUDGE.format(bullets="\n".join(f"- {b}" for b in bullets), concept=concept)
    txt, err = llm_call(msg, MODEL, max_tokens=1024, temperature=0.0)
    if txt is None:
        return None, f"JUDGE_ERROR:{err}"
    try:
        i = txt.find("{")
        obj, _ = json.JSONDecoder().raw_decode(txt[i:])
        return bool(obj.get("present")), str(obj.get("evidence", ""))[:160]
    except Exception:
        return None, "JUDGE_ERROR:parse"


def load_dataset(name):
    p = HERE / "datasets" / f"{name}.json"
    return json.load(open(p))["items"] if p.exists() else []


def build_queries(items, readouts, n_decoy, ks, seed=0):
    """One query per (item, intermediate, k, true|decoy). Decoys are drawn from the same
    distribution so they are equally plausible a priori -- a decoy from another distribution
    would be trivially rejectable and would flatter the read-out."""
    rng = random.Random(seed)
    pool = [a for it in items for i in it["intermediates"] for a in i["accept"]]
    qs = []
    for idx, it in enumerate(items):
        b = readouts.get(str(idx))
        if not b:
            continue
        for spec in it["intermediates"]:
            true_c = spec["accept"][0]
            others = [c for c in pool if c.lower() not in
                      {a.lower() for a in spec["accept"]}]
            decoys = rng.sample(others, min(n_decoy, len(others)))
            for k in ks:
                if k > len(b):
                    continue
                cand = [(true_c, True)] + [(d, False) for d in decoys]
                rng.shuffle(cand)                      # no positional habit can mimic signal
                for c, is_true in cand:
                    qs.append({"item": idx, "field": spec["name"], "k": k,
                               "concept": c, "is_true": is_true, "bullets": b[:k]})
    return qs


def run(qs, workers=12):
    def one(q):
        p, ev = judge_one(q["bullets"], q["concept"])
        return {**q, "present": p, "evidence": ev}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, qs))


def summarise(res, ks):
    out = {}
    for k in ks:
        r = [x for x in res if x["k"] == k and x["present"] is not None]
        t = [x for x in r if x["is_true"]]
        d = [x for x in r if not x["is_true"]]
        hit = sum(x["present"] for x in t) / max(len(t), 1)
        fp = sum(x["present"] for x in d) / max(len(d), 1)
        out[k] = {"hit": hit, "decoy_fp": fp, "discrimination": hit - fp,
                  "n_true": len(t), "n_decoy": len(d)}
    # paper's summary: AUC of pass@k vs log k, normalised so all-recovered-at-k=1 -> 1.0
    import math
    xs = [math.log(k) for k in ks]
    for key in ("hit", "discrimination"):
        ys = [out[k][key] for k in ks]
        if len(ks) > 1:
            auc = sum((ys[i] + ys[i + 1]) / 2 * (xs[i + 1] - xs[i])
                      for i in range(len(ks) - 1)) / (xs[-1] - xs[0])
        else:
            auc = ys[0]
        out[f"auc_{key}"] = auc
    return out


def selftest():
    """Validate the JUDGE before trusting it on real read-outs, using synthetic read-outs whose
    answer is known: bullets that state the concept (must pass), bullets from an unrelated item
    (must fail), and vague filler (must fail). A judge that cannot separate these cannot be
    trusted to score a lens."""
    cases = [
        (["the planet Mars is reddish", "fourth from the sun"], "Mars", True, "explicit"),
        (["a recipe for lemon cake", "preheat the oven"], "Mars", False, "unrelated"),
        (["something is happening", "there is a thing", "it continues"], "Mars", False, "vague"),
        (["the red planet", "iron oxide dust storms"], "Mars", True, "implied, not named"),
        (["she misses him terribly", "his mug is still there"], "grief", True, "implied"),
        (["she misses him terribly", "his mug is still there"], "jealousy", False, "near-miss"),
        (["multiplication comes after the bracket", "then times four"],
         "multiplication", True, "explicit"),
        (["addition of two numbers", "sum them first"], "multiplication", False, "near-miss"),
    ]
    print(f"{'expect':>7}{'got':>6}  {'why':<20} concept")
    ok = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        got = list(ex.map(lambda c: judge_one(c[0], c[1]), cases))
    for (b, c, want, why), (p, ev) in zip(cases, got):
        good = p == want
        ok += good
        print(f"{str(want):>7}{str(p):>6}  {why:<20} {c}{'' if good else '   <-- MISMATCH'}")
    print(f"\njudge agreement with known ground truth: {ok}/{len(cases)}")
    print("near-miss cases are the ones that matter: a judge that passes 'jealousy' on a grief")
    print("passage, or 'multiplication' on an addition phrase, will inflate every hit rate.")
    return ok == len(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readouts", help="JSON: {distribution: {item_idx: [bullets]}}")
    ap.add_argument("--dists", default=",".join(DISTS))
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--n_decoy", type=int, default=2)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--selftest", action="store_true")
    A = ap.parse_args()

    if A.selftest:
        raise SystemExit(0 if selftest() else 1)
    assert A.readouts, "pass --readouts, or --selftest to validate the judge alone"
    R = json.load(open(A.readouts))
    ks = [int(v) for v in A.ks.split(",")]
    all_res, summary = [], {}
    for name in A.dists.split(","):
        items = load_dataset(name)
        if not items or name not in R:
            print(f"[skip] {name}: {'no dataset' if not items else 'no read-outs'}")
            continue
        qs = build_queries(items, R[name], A.n_decoy, ks)
        print(f"[judge] {name}: {len(qs)} judgements ...", flush=True)
        res = run(qs)
        for x in res:
            x["dist"] = name
        all_res += res
        summary[name] = summarise(res, ks)

    print(f"\n{'distribution':<15}{'k':>3}{'hit':>8}{'decoy FP':>10}{'discrim':>9}")
    for name, s in summary.items():
        for k in ks:
            r = s[k]
            print(f"{name:<15}{k:>3}{r['hit']:>8.3f}{r['decoy_fp']:>10.3f}"
                  f"{r['discrimination']:>+9.3f}")
        print(f"{'':<15}{'AUC':>3}{s['auc_hit']:>8.3f}{'':>10}{s['auc_discrimination']:>+9.3f}")
    if summary:
        md = sum(s["auc_discrimination"] for s in summary.values()) / len(summary)
        print(f"\nmean discrimination AUC across distributions: {md:+.3f}")
        print("0 means the judge assents to decoys as often as to true intermediates, i.e. the")
        print("read-out carries nothing about the intermediate regardless of its hit rate.")
    dst = HERE / "results"
    dst.mkdir(exist_ok=True)
    json.dump({"tag": A.tag, "model": MODEL, "ks": ks, "n_decoy": A.n_decoy,
               "summary": summary, "judgements": all_res},
              open(dst / f"{A.tag}.json", "w"), indent=1)
    print(f"saved {dst}/{A.tag}.json")


if __name__ == "__main__":
    main()
