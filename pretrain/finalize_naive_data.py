"""Naive future-lens variant of finalize_ao_data.py: identical pipeline, but the SFT
target is a RAW model continuation (rollouts[i]) instead of the Claude `description`.
Run on the SAME per-layer train_L{l}.parquet the Claude AO uses, so positions,
activations, and the doc-disjoint split are IDENTICAL — the label is the only
difference (Claude "thoughts" vs raw next tokens). That makes claude-joracle vs
futurelens-joracle a fully controlled comparison."""
import argparse, json, datetime
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ACTOR_TEMPLATE = (
    "You are shown an internal activation vector captured from a language model "
    "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
    "taken at one position and encodes what the model is about to generate next. "
    "Output the text the model most likely produces immediately after this point.\n\n"
    "<concept>{injection_char}</concept>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, help="per-layer train parquet with a `rollouts` column")
    ap.add_argument("--meta", required=True, help="collection raw.parquet.meta.json")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--rollout-idx", type=int, default=0, help="which rollout to use as the raw target")
    args = ap.parse_args()

    t = pq.read_table(args.labeled)
    d = t.to_pylist()
    kept = []
    for r in d:
        rolls = r.get("rollouts") or []
        # first non-empty rollout at/after rollout-idx -> the raw next-token target
        resp = next((x.strip() for x in rolls[args.rollout_idx:] + rolls if x and x.strip()), "")
        if not resp:
            continue
        r["response"] = resp
        kept.append(r)
    d = kept
    print(f"{len(d)} rows with a raw-continuation target")

    docs = sorted({r["doc_id"] for r in d})
    nval = max(1, int(len(docs) * args.val_frac))
    valset = set(docs[:nval])
    ps = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
    def to_tbl(rows):
        return pa.table({
            "prompt": pa.array([r["prompt"] for r in rows], type=ps),
            "response": pa.array([r["response"] for r in rows], type=pa.string()),
            "activation_vector": pa.array([r["activation_vector"] for r in rows], type=pa.list_(pa.float32())),
            "ctx_text": pa.array([r.get("ctx_text", "") for r in rows], type=pa.string()),
            "doc_id": pa.array([r["doc_id"] for r in rows], type=pa.string()),
        })
    train = to_tbl([r for r in d if r["doc_id"] not in valset])
    val = to_tbl([r for r in d if r["doc_id"] in valset])
    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(train, args.out_train, row_group_size=2000)
    pq.write_table(val, args.out_val, row_group_size=1000)
    print(f"train={train.num_rows} val={val.num_rows}")

    m = json.loads(Path(args.meta).read_text())
    sidecar = {
        "dataset_id": f"naive_fl_{args.base_model.split('/')[-1]}_L{m['layer']}",
        "stage": "av_sft", "row_count": train.num_rows, "kind": "nla_dataset",
        "schema_version": 1, "keep_debug_metadata": True,
        "extraction": {"base_model": args.base_model, "d_model": m["d_model"],
                       "layer_index": m["layer"], "norm": "none"},
        "tokens": {"injection_char": m["inj_char"], "injection_token_id": m["inj_id"],
                   "injection_left_neighbor_id": m["left"], "injection_right_neighbor_id": m["right"],
                   "critic_suffix_ids": None},
        "prompt_templates": {"actor": ACTOR_TEMPLATE},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": "scripts.finalize_naive_data",
    }
    for o in (args.out_train, args.out_val):
        Path(o + ".nla_meta.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False, allow_unicode=True))
    print("wrote sidecars")


if __name__ == "__main__":
    main()
