"""Can we build span-conditional averages from data already on disk?

The proposal (and the paper's template-lens method): condition on the SPAN and
average over CONTEXTS. For a span S, average h42 across many rows whose
continuation is S, giving a general-disposition vector that still has ground-truth
verbalization. Both train and test vectors are then averages over contexts, so the
train/test gap shrinks from "one context vs all contexts" to "contexts that say S
vs all contexts".

Whether that is free depends entirely on span REPETITION: an exact-match group
needs the same span to occur many times. Longer spans are more unique, so this
measures the tradeoff directly — for each span length, how many rows live in a
group of size >= k.

If 8-token spans almost never repeat (likely), the options are: shorten the span
until grouping works, group by semantic similarity instead of exact match, or
generate fresh contexts per span the way the paper does.
"""
import argparse
import collections
import glob
import json

import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="/workspace/data/spans_jvp/*_jvp.parquet")
ap.add_argument("--lengths", default="1,2,3,4,6,8")
ap.add_argument("--out", default="/workspace/results/span_repetition.json")
args = ap.parse_args()

files = sorted(f for f in glob.glob(args.shards) if not f.endswith("_probe.parquet"))
LENS = [int(x) for x in args.lengths.split(",")]
counters = {L: collections.Counter() for L in LENS}
n_rows = 0
for f in files:
    pf = pq.ParquetFile(f)
    for rg in range(pf.num_row_groups):
        for r in pf.read_row_group(rg, columns=["rollout_token_ids"]).to_pylist():
            roll = r["rollout_token_ids"]
            if not roll or len(roll) < max(LENS):
                continue
            n_rows += 1
            for L in LENS:
                counters[L][tuple(int(t) for t in roll[:L])] += 1
print(f"{n_rows} rows over {len(files)} shards\n")

res = {"n_rows": n_rows, "by_length": {}}
print(f"{'span len':>8} {'distinct':>10} {'mean grp':>9} "
      + "".join(f"{'rows in grp>=' + str(k):>16}" for k in (2, 5, 10, 25)))
for L in LENS:
    c = counters[L]
    tot = sum(c.values())
    ent = {"distinct": len(c), "mean_group": tot / max(1, len(c))}
    row = f"{L:>8} {len(c):>10} {ent['mean_group']:>9.2f}"
    for k in (2, 5, 10, 25):
        rows_in = sum(v for v in c.values() if v >= k)
        ent[f"rows_in_groups_ge_{k}"] = rows_in
        ent[f"frac_ge_{k}"] = rows_in / max(1, tot)
        row += f"{rows_in:>10} ({100 * rows_in / max(1, tot):4.1f}%)"
    res["by_length"][L] = ent
    print(row)

print("\nmost common spans (len 4), decoded later:")
for sp, n in counters[min(4, max(LENS))].most_common(8):
    print(f"   x{n:5d}  {list(sp)}")

best = max(LENS, key=lambda L: res["by_length"][L]["frac_ge_5"])
print(f"\nbest length for exact-match grouping: {best} "
      f"({100 * res['by_length'][best]['frac_ge_5']:.1f}% of rows in a group of >=5)")
print("VERDICT:", "exact-match span grouping is viable on existing data"
      if res["by_length"][max(LENS)]["frac_ge_5"] > 0.10 else
      "8-token spans are too unique for exact-match grouping — need shorter spans, "
      "semantic grouping, or fresh per-span context generation")
json.dump(res, open(args.out, "w"), indent=1)
print(f"wrote {args.out}")
