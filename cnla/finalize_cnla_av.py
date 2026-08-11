"""AV-SFT data for the compositional NLA: target = the full 4-bullet doc.

Reads the collect+format parquet (cnla_L62.parquet: prompt chat-list with the inject
marker, activation_vector = L62 activation, bullets_text = "* c1\n* c2\n* c3\n* c4")
and writes a train/val SFT parquet + sidecar in the schema train_sft --mode av expects
(prompt, response, activation_vector, ctx_text, doc_id). The response is the whole
bullet block so the RL policy learns to emit the 4-bullet format from an injected
activation. Doc-disjoint split (same doc never straddles train/val).
"""
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
    ap.add_argument("--data", required=True, help="cnla_L62.parquet (has bullets_text)")
    ap.add_argument("--meta", required=True, help="collect meta.json (inj_char/id/neighbors, d_model, layer)")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--min-bullets", type=int, default=2, help="drop rows with fewer parseable bullets")
    args = ap.parse_args()

    d = pq.read_table(args.data).to_pylist()
    kept = []
    for r in d:
        bt = (r.get("bullets_text") or "").strip()
        if bt.count("*") < args.min_bullets:   # need at least a couple of bullets
            continue
        r["response"] = bt
        kept.append(r)
    d = kept
    print(f"{len(d)} rows with a >= {args.min_bullets}-bullet target")

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
    for o in (args.out_train, args.out_val):
        Path(o).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(train, args.out_train, row_group_size=2000)
    pq.write_table(val, args.out_val, row_group_size=1000)
    print(f"train={train.num_rows} val={val.num_rows}")

    m = json.loads(Path(args.meta).read_text())
    sidecar = {
        "dataset_id": f"cnla_av_{args.base_model.split('/')[-1]}_L{m['layer']}",
        "stage": "av_sft", "row_count": train.num_rows, "kind": "nla_dataset",
        "schema_version": 1, "keep_debug_metadata": True,
        "extraction": {"base_model": args.base_model, "d_model": m["d_model"],
                       "layer_index": m["layer"], "norm": "none"},
        "tokens": {"injection_char": m["inj_char"], "injection_token_id": m["inj_id"],
                   "injection_left_neighbor_id": m["left"], "injection_right_neighbor_id": m["right"],
                   "critic_suffix_ids": None},
        "prompt_templates": {"actor": ACTOR_TEMPLATE},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": "cnla.finalize_cnla_av",
    }
    for o in (args.out_train, args.out_val):
        Path(o + ".nla_meta.yaml").write_text(yaml.safe_dump(sidecar, sort_keys=False, allow_unicode=True))
    print("wrote sidecars")


if __name__ == "__main__":
    main()
