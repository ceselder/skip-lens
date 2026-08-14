"""Build token-exact data for future-lens SFT and on-policy distillation.

The output deliberately keeps three distinct views of a training example:

* ``activation_vector``: all the student is allowed to observe;
* ``teacher_input_ids``: the real source prefix, visible only to the teacher;
* ``target_ids``: one sampled base-model rollout, used by the matched SFT arm.

The OPD arm does not train on ``target_ids``.  It samples its own prefix and is
distilled from the base model conditioned on ``teacher_input_ids`` plus that
sampled prefix.  Keeping the fields separate makes accidental context leakage
into the student easy to test and audit.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from transformers import AutoTokenizer

from pretrain.finalize_naive_data import ACTOR_TEMPLATE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collected", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--rollout-idx", type=int, default=0)
    ap.add_argument("--max-target-tokens", type=int, default=16)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    table = pq.read_table(args.collected)
    activation_columns = [name for name in table.column_names if name.startswith("act_L")]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    required = {
        "prompt", "activation_vector", "teacher_input_ids",
        "rollout_token_ids", "continuation_ids", "doc_id", "ctx_text",
    }
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(
            f"{args.collected} predates OPD collection and is missing {sorted(missing)}; "
            "re-run pretrain.collect_ao_data"
        )

    rows = []
    for row in table.to_pylist():
        candidates = row["rollout_token_ids"] or []
        order = candidates[args.rollout_idx:] + candidates[:args.rollout_idx]
        target = next((x for x in order if x), None)
        if not target:
            continue
        row["target_ids"] = target[:args.max_target_tokens]
        row["response"] = tokenizer.decode(row["target_ids"], skip_special_tokens=True)
        row["continuation_ids"] = (row["continuation_ids"] or [])[:args.max_target_tokens]
        rows.append(row)

    docs = sorted(
        {r["doc_id"] for r in rows},
        key=lambda doc: hashlib.sha256(
            f"{args.split_seed}:{doc}".encode()
        ).digest(),
    )
    n_val = max(1, round(len(docs) * args.val_frac))
    val_docs = set(docs[:n_val])
    prompt_type = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))

    def make_table(items):
        columns = {
            "prompt": pa.array([r["prompt"] for r in items], type=prompt_type),
            "response": pa.array([r["response"] for r in items], type=pa.string()),
            "activation_vector": pa.array(
                [r["activation_vector"] for r in items], type=pa.list_(pa.float32())),
            "teacher_input_ids": pa.array(
                [r["teacher_input_ids"] for r in items], type=pa.list_(pa.int32())),
            "target_ids": pa.array(
                [r["target_ids"] for r in items], type=pa.list_(pa.int32())),
            "continuation_ids": pa.array(
                [r["continuation_ids"] for r in items], type=pa.list_(pa.int32())),
            "ctx_text": pa.array([r["ctx_text"] for r in items], type=pa.string()),
            "doc_id": pa.array([r["doc_id"] for r in items], type=pa.string()),
        }
        for name in activation_columns:
            columns[name] = pa.array([r[name] for r in items], type=pa.list_(pa.float32()))
        return pa.table(columns)

    train = make_table([r for r in rows if r["doc_id"] not in val_docs])
    val = make_table([r for r in rows if r["doc_id"] in val_docs])
    for path, split in ((args.out_train, train), (args.out_val, val)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(split, path, row_group_size=2000)

    meta = json.loads(Path(args.meta).read_text())
    sidecar = {
        "dataset_id": f"opd_{args.base_model.split('/')[-1]}_L{meta['layer']}",
        "stage": "opd", "row_count": train.num_rows,
        "kind": "nla_dataset", "schema_version": 1,
        "extraction": {
            "base_model": args.base_model, "d_model": meta["d_model"],
            "layer_index": meta["layer"], "norm": "none",
        },
        "generation": {
            "corpus": meta.get("corpus"),
            "corpus_config": meta.get("corpus_config"),
            "temperature": meta.get("rollout_temperature"),
            "top_p": meta.get("rollout_top_p"),
            "rollout_len": meta.get("rollout_len"),
            "decision_points": meta.get("decision_points"),
            "rollout_index": args.rollout_idx,
        },
        "tokens": {
            "injection_char": meta["inj_char"],
            "injection_token_id": meta["inj_id"],
            "injection_left_neighbor_id": meta["left"],
            "injection_right_neighbor_id": meta["right"],
            "critic_suffix_ids": None,
        },
        "prompt_templates": {"actor": ACTOR_TEMPLATE},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": "pretrain.finalize_opd_data",
    }
    for path in (args.out_train, args.out_val):
        Path(path + ".nla_meta.yaml").write_text(
            yaml.safe_dump(sidecar, sort_keys=False, allow_unicode=True))
    print(f"train={train.num_rows} val={val.num_rows}; target horizon <= {args.max_target_tokens}")


if __name__ == "__main__":
    main()
