"""Per-condition breakdown of the judged multi-slot eval.

judge_fedlayer.py pools every record it is handed, so the 5 arm-A conditions
(per_offset + 4 knockouts) collapse into one number. This splits them back
apart on the `condition` field and prints workspace/answer agreement per
condition, plus a few verbatim readouts for the vibe check.
"""
import argparse
import collections
import json
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--judged", required=True)
ap.add_argument("--out", default="")
ap.add_argument("--examples", type=int, default=3)
args = ap.parse_args()

d = json.load(open(args.judged))
recs = (d if isinstance(d, list)
        else d.get("detail") or d.get("records") or d.get("items") or [])
print(f"n records: {len(recs)}")
print(f"keys: {sorted(recs[0].keys())}")

W_KEYS = ("agree_jlens", "jlens_agree", "workspace", "score_jlens", "jlens")
A_KEYS = ("agree_answer", "answer_agree", "answer", "score_answer")


def pick(r, keys):
    for k in keys:
        if k in r and isinstance(r[k], (int, float)):
            return float(r[k])
    return None


by = collections.defaultdict(lambda: {"w": [], "a": []})
for r in recs:
    c = r.get("condition", "?")
    w, a = pick(r, W_KEYS), pick(r, A_KEYS)
    if w is not None:
        by[c]["w"].append(w)
    if a is not None:
        by[c]["a"].append(a)

rows = []
for c, v in sorted(by.items(), key=lambda x: -(st.mean(x[1]["w"]) if x[1]["w"] else 0)):
    if not v["w"]:
        continue
    w, a = st.mean(v["w"]), (st.mean(v["a"]) if v["a"] else float("nan"))
    n = len(v["w"])
    se = (st.stdev(v["w"]) / (n ** 0.5)) if n > 1 else 0.0
    rows.append({"condition": c, "workspace": w, "answer": a, "n": n, "se": se})
    print(f"{c:20s} workspace={w:.3f}±{se:.3f}  answer={a:.3f}  n={n}")

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
