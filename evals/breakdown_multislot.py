"""Per-condition breakdown of the judged multi-slot eval.

SCALE WARNING. The judge emits 0/1/2 per record. judge_fedlayer.py's own summary
prints those DIVIDED BY 2 (normalized 0-1); this script used to print the raw
0-2 mean. Reading the baseline off one and the arms off the other understated
the baseline by 2x and produced hours of bogus "reaches parity" claims. Both
scales are now printed, normalized first, and `norm` is what the JSON carries
as the primary field.

judge_fedlayer.py pools every record it is handed, so the 5 arm-A conditions
(per_offset + 4 knockouts) collapse into one number. This splits them back
apart on the `condition` field and prints workspace/answer agreement per
condition, plus a few verbatim readouts for the vibe check.
"""
import argparse
import collections
import json
import statistics as st

def is_degenerate(text):
    """A collapsed readout, not a readout: a short unit repeated to fill the
    span ('88888888', 'aaaa'), or <=2 distinct words over >=4 words. The frozen
    arm scored 0.439 with 47% of its outputs like this, so any judged mean is
    meaningless without this number beside it."""
    t = (text or "").strip()
    if len(t) < 4:
        return True
    for n in (1, 2, 3):
        reps = len(t) // n
        if reps >= 4 and t[: n * reps] == t[:n] * reps:
            return True
    w = t.split()
    return len(w) >= 4 and len(set(w)) <= 2


ap = argparse.ArgumentParser()
ap.add_argument("--judged", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--examples", type=int, default=3)
args = ap.parse_args()

d = json.load(open(args.judged))
recs = (d if isinstance(d, list)
        else d.get("detail") or d.get("records") or d.get("items") or [])
print(f"n records: {len(recs)}")
print("scores below are NORMALIZED 0-1 (judge emits 0/1/2); raw in parens")
print(f"keys: {sorted(recs[0].keys())}")



W_KEYS = ("agree_jlens", "jlens_agree", "workspace", "score_jlens", "jlens")
A_KEYS = ("agree_answer", "answer_agree", "answer", "score_answer")


def pick(r, keys):
    for k in keys:
        if k in r and isinstance(r[k], (int, float)):
            return float(r[k])
    return None


by = collections.defaultdict(lambda: {"w": [], "a": [], "deg": 0, "n_out": 0,
                                      "hist": collections.Counter()})
for r in recs:
    c = r.get("condition", "?")
    w, a = pick(r, W_KEYS), pick(r, A_KEYS)
    if w is not None:
        by[c]["w"].append(w)
        by[c]["hist"][int(w)] += 1
    if a is not None:
        by[c]["a"].append(a)
    ro = r.get("readout")
    ro = ro[0] if isinstance(ro, list) and ro else ro
    if ro is not None:
        by[c]["n_out"] += 1
        by[c]["deg"] += int(is_degenerate(ro))

rows = []
for c, v in sorted(by.items(), key=lambda x: -(st.mean(x[1]["w"]) if x[1]["w"] else 0)):
    if not v["w"]:
        continue
    w, a = st.mean(v["w"]), (st.mean(v["a"]) if v["a"] else float("nan"))
    n = len(v["w"])
    se = (st.stdev(v["w"]) / (n ** 0.5)) if n > 1 else 0.0
    dg = v["deg"] / v["n_out"] if v["n_out"] else float("nan")
    rows.append({"condition": c,
                 "workspace": w / 2, "answer": a / 2,          # normalized 0-1
                 "workspace_raw02": w, "answer_raw02": a,      # raw judge scale
                 "n": n, "se": se / 2, "se_raw02": se,
                 "degenerate_frac": dg, "score_hist": dict(v["hist"])})
    flag = "  <-- MOSTLY DEGENERATE" if dg > 0.25 else ""
    print(f"{c:20s} workspace={w/2:.3f}±{se/2:.3f} (raw {w:.3f})  "
          f"answer={a/2:.3f}  n={n}  degenerate={100*dg:.0f}%  hist(0/1/2)="
          f"{v['hist'][0]}/{v['hist'][1]}/{v['hist'][2]}{flag}")

ex = []
for r in recs:
    if r.get("condition") == "per_offset" and len(ex) < args.examples:
        ex.append({"name": r.get("name"), "condition": r.get("condition"),
                   "context": (r.get("context") or "")[-160:],
                   "readout": (r.get("readout") or [""])[0] if isinstance(r.get("readout"), list) else r.get("readout"),
                   "jlens_top": r.get("jlens_top"), "actual": r.get("actual")})
print("\n--- verbatim per_offset examples ---")
for e in ex:
    print(f"\n[{e['name']}] …{e['context']}")
    print(f"  READOUT: {e['readout']}")
    print(f"  J-lens:  {e['jlens_top']}")

if args.out:
    json.dump({"rows": rows, "examples": ex}, open(args.out, "w"),
              ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")
