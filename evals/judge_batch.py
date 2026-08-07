"""Batch-API judge over ALL naive-variant readouts -> cross-variant discrimination table.

~50k judgements (48 configs x 6 datasets x items x {1 true + n_decoy} x ks), which is squarely a
Message Batches job (50% cheaper, no rate-limit thrash) rather than synchronous calls. The grading
logic is IDENTICAL to judge_intermediates.py -- same JUDGE prompt, same decoy construction
(discrimination = hit - decoy-FP, so an agreeable grader scores ~0), same summary. Only the
transport differs: one async batch instead of a ThreadPool of sync calls.

Key: $ANTHROPIC_API_KEY_BATCH (batch endpoint). Model: claude-sonnet-5.

    python evals/judge_batch.py --results "evals/results/ro__*.json" --ks 8 --n-decoy 2 \
        --out evals/results/judge_table.json [--validate 20]
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import anthropic

from judge_intermediates import JUDGE, MODEL, DISTS, load_dataset, build_queries, summarise

HERE = Path(__file__).parent


def _variant_tag(path):
    return os.path.basename(path).replace("ro__", "").replace(".json", "")   # e.g. naive_fl_L62__jac42


def build_requests(results_glob, ks, n_decoy):
    """One request per (config, dataset, query). custom_id is an index; meta holds the rest."""
    reqs, meta = [], {}
    files = sorted(f for f in glob.glob(results_glob) if "ro_smoke" not in f)
    for f in files:
        tag = _variant_tag(f)
        R = json.load(open(f))
        for dist in DISTS:
            items = load_dataset(dist)
            if not items or dist not in R:
                continue
            for q in build_queries(items, R[dist], n_decoy, ks):
                cid = f"q{len(reqs)}"
                prompt = JUDGE.format(bullets="\n".join(f"- {b}" for b in q["bullets"]),
                                      concept=q["concept"])
                reqs.append({"custom_id": cid,
                             "params": {"model": MODEL, "max_tokens": 1024,
                                        "messages": [{"role": "user", "content": prompt}]}})
                meta[cid] = {"tag": tag, "dist": dist, "is_true": bool(q["is_true"]), "k": int(q["k"])}
    return reqs, meta, files


def _present(text):
    try:
        i = text.find("{")
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
        return bool(obj.get("present"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results" / "ro__*.json"))
    ap.add_argument("--ks", default="8")
    ap.add_argument("--n-decoy", type=int, default=2)
    ap.add_argument("--out", default=str(HERE / "results" / "judge_table.json"))
    ap.add_argument("--validate", type=int, default=0, help="submit only the first N requests (smoke)")
    ap.add_argument("--poll", type=int, default=30, help="seconds between batch status polls")
    A = ap.parse_args()
    ks = [int(v) for v in A.ks.split(",")]

    key = os.environ.get("ANTHROPIC_API_KEY_BATCH")
    assert key, "set $ANTHROPIC_API_KEY_BATCH"
    client = anthropic.Anthropic(api_key=key)

    reqs, meta, files = build_requests(A.results, ks, A.n_decoy)
    if A.validate:
        reqs = reqs[:A.validate]
    print(f"[judge_batch] {len(files)} configs, {len(reqs)} judgements "
          f"(ks={ks}, n_decoy={A.n_decoy}){' [VALIDATE]' if A.validate else ''}", flush=True)

    batch = client.messages.batches.create(requests=reqs)
    print(f"[judge_batch] submitted batch {batch.id}; polling ...", flush=True)
    while True:
        b = client.messages.batches.retrieve(batch.id)
        if b.processing_status == "ended":
            break
        c = b.request_counts
        print(f"  [{b.processing_status}] done={c.succeeded} err={c.errored} "
              f"proc={c.processing}", flush=True)
        time.sleep(A.poll)

    present = {}
    for r in client.messages.batches.results(batch.id):
        if r.result.type == "succeeded":
            txt = "\n".join(bl.text for bl in r.result.message.content
                            if getattr(bl, "type", "") == "text")
            present[r.custom_id] = _present(txt)
        else:
            present[r.custom_id] = None

    # regroup into judge_intermediates' per-item records, then reuse summarise() per (config,dist)
    by_cfg = {}
    for cid, m in meta.items():
        p = present.get(cid)
        by_cfg.setdefault(m["tag"], {}).setdefault(m["dist"], []).append(
            {"k": m["k"], "is_true": m["is_true"], "present": p})

    table = {}
    for tag, dists in by_cfg.items():
        per_dist = {d: summarise(recs, ks) for d, recs in dists.items()}
        md = sum(s["auc_discrimination"] for s in per_dist.values()) / max(len(per_dist), 1)
        mh = sum(s["auc_hit"] for s in per_dist.values()) / max(len(per_dist), 1)
        table[tag] = {"mean_disc_auc": md, "mean_hit_auc": mh, "per_dist": per_dist}

    out = {"model": MODEL, "ks": ks, "n_decoy": A.n_decoy, "batch_id": batch.id,
           "n_judgements": len(reqs), "table": table}
    Path(os.path.dirname(A.out)).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(A.out, "w"), indent=1)

    rows = sorted(table.items(), key=lambda kv: kv[1]["mean_disc_auc"], reverse=True)
    print(f"\n{'config':<28}{'disc_AUC':>10}{'hit_AUC':>9}")
    for tag, s in rows:
        print(f"{tag:<28}{s['mean_disc_auc']:>+10.3f}{s['mean_hit_auc']:>9.3f}")
    print(f"\nsaved {A.out}   (discrimination = hit - decoy-FP; ~0 means the readout carries "
          "nothing about the intermediate)")


if __name__ == "__main__":
    main()
