"""Arm I — token-pooled, context-UNpooled training vector; averaged at test.

    TRAIN  v = [ sum_{d} J_local^(d)(this context) ] @ h42[p]      ONE vector
    TEST   v = [ sum_{d} Jbar^(d) ] @ h42[p]   ( = the paper's J-lens vector )

Both sides are "a token-pooled Jacobian applied to h42", so the ONLY difference
is whether the Jacobian was averaged over contexts — which is the skip-lens
thesis itself. Every earlier arm also changed the object TYPE (derivative vs
activation vs per-horizon transport), and that confound is what sent four
mechanistic hypotheses down blind alleys.

Why one vector rather than K slots, from tonight's measurements:
  * per-slot norm-matching amplifies the deep slots ~500x at test time (their
    estimates carry 0.2-3% of the real state's norm) -> +5.80 nats of collapse;
  * the only configuration in this study that produces decent readouts is K=1
    (armC, 0.561), and it already emits 8 tokens from a single vector.

The training vector is free: pass 2 stored the 16 per-offset LOCAL transports
per row, so summing them gives the token-pooled local transport (truncated to
the fitted window, which the audit measured at 0.93 cosine with a full pooled
fit). The response is the same 8-token on-policy span the other arms use.
"""
import argparse
import glob
import hashlib
import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SINGLE_TEMPLATE = (
    "You are shown an internal activation vector captured from a language model "
    "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
    "taken at one position and encodes what the model is about to generate over "
    "the next several tokens. Output the text the model most likely produces "
    "immediately after this point.\n\n<concept>{injection_char}</concept>")

INJECT_PLACEHOLDER = "<INJECT>"
PS = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
OUT_SCHEMA = pa.schema([
    ("prompt", PS), ("response", pa.string()),
    ("activation_vector", pa.list_(pa.float32())),
    ("ctx_text", pa.string()), ("doc_id", pa.string()), ("span_len", pa.int32()),
])


def is_val(doc_id, frac):
    return int(hashlib.md5(str(doc_id).encode()).hexdigest(), 16) % 10000 < int(frac * 10000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-glob", default="/workspace/data/spans_jvp/shard_[0-3]_jvp.parquet",
                    help="pass-2 output holding the 16 per-offset LOCAL transports")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--span-len", type=int, default=8,
                    help="response length in tokens (the pooled vector is asked "
                         "to support this many)")
    ap.add_argument("--pool-upto", type=int, default=16,
                    help="sum local transports over d < this")
    ap.add_argument("--deep-from", type=int, default=0,
                    help="start the sum at this offset. Setting it to 1 excludes "
                         "the current-token term, which is the only offset where "
                         "raw h42 already beats the Jacobian (0.437 vs 0.395) and "
                         "whose weight dominates a uniform sum since "
                         "||Jbar^(0)||=30.8 against 6.6, 3.4, ... Excluding it is "
                         "what makes the averaged Jacobian additive over h42: "
                         "+0.074 alignment margin, and +0.044 vs +0.019 on "
                         "context-specific future-token recall.")
    ap.add_argument("--weight-mode", choices=["uniform", "norm_eq"], default="uniform",
                    help="norm_eq sets w_d = 1/||Jbar^(d)|| so every horizon "
                         "contributes comparably instead of offset 0 swamping the "
                         "rest. The SAME weights are used to build the test vector, "
                         "so this is a design choice, not a train/test mismatch.")
    ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens",
                    help="only read for its matrix norms under --weight-mode norm_eq")
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--min-h42-cos", type=float, default=0.99)
    args = ap.parse_args()

    W = np.zeros(16, dtype=np.float32)
    if args.weight_mode == "norm_eq":
        for d in range(args.deep_from, args.pool_upto):
            J = np.load(f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy", mmap_mode="r")
            W[d] = 1.0 / float(np.linalg.norm(J))
    else:
        W[args.deep_from:args.pool_upto] = 1.0
    print("pool weights:", " ".join(f"{x:.4f}" for x in W), flush=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    files = sorted(f for f in glob.glob(args.shards_glob)
                   if not f.endswith("_probe.parquet"))
    assert files, f"no shards matched {args.shards_glob}"
    prompt_msgs = [{"role": "user",
                    "content": SINGLE_TEMPLATE.format(injection_char=INJECT_PLACEHOLDER)}]
    writers, n = {}, {"train": 0, "val": 0, "cos": 0, "short": 0, "roundtrip": 0}

    def wr(p):
        if p not in writers:
            writers[p] = pq.ParquetWriter(p + ".tmp", OUT_SCHEMA)
        return writers[p]

    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            out = {args.out_train: [], args.out_val: []}
            for r in pf.read_row_group(rg).to_pylist():
                if r["h42_recompute_cosine"] < args.min_h42_cos:
                    n["cos"] += 1
                    continue
                roll = r["rollout_token_ids"]
                if len(roll) < args.span_len:
                    n["short"] += 1
                    continue
                resp = tok.decode(roll[: args.span_len], skip_special_tokens=True)
                if len(resp.strip()) < 2:
                    n["short"] += 1
                    continue
                if tok.encode(resp, add_special_tokens=False) != [
                        int(x) for x in roll[: args.span_len]]:
                    n["roundtrip"] += 1
                    continue
                tv = np.frombuffer(r["transported_vectors"], dtype=np.float16)
                tv = tv.reshape(16, -1).astype(np.float32)
                v = (W[:, None] * tv).sum(0)       # token-pooled LOCAL transport
                dest = args.out_val if is_val(r["doc_id"], args.val_frac) else args.out_train
                out[dest].append({
                    "prompt": prompt_msgs, "response": resp,
                    "activation_vector": v.tolist(),
                    "ctx_text": r["ctx_text"], "doc_id": r["doc_id"],
                    "span_len": args.span_len})
            for path, batch in out.items():
                if not batch:
                    continue
                wr(path).write_table(pa.table({
                    nm: pa.array([b[nm] for b in batch], type=OUT_SCHEMA.field(nm).type)
                    for nm in OUT_SCHEMA.names}))
                n["train" if path == args.out_train else "val"] += len(batch)
        print(f"  {os.path.basename(f)}: {n}", flush=True)

    for path, w in writers.items():
        w.close()
        os.replace(path + ".tmp", path)

    import datetime

    import yaml

    from nla.datagen.injection_tokens import find_injection_token
    from nla.schema import compute_canonical_neighbors
    ic, iid = find_injection_token(tok)
    left, right = compute_canonical_neighbors(tok, SINGLE_TEMPLATE, ic, iid)
    side = {"dataset_id": f"pooled_single_L42to62_span{args.span_len}",
            "stage": "av_sft", "row_count": n["train"], "kind": "nla_dataset",
            "schema_version": 1, "keep_debug_metadata": True, "n_slots": 1,
            "extraction": {"base_model": args.base_model, "d_model": 5120,
                           "layer_index": 62, "source_layer": 42,
                           "transport": (f"token_pooled_LOCAL_jacobian_"
                                         f"{args.weight_mode}_d{args.deep_from}"
                                         f"to{args.pool_upto}"),
                           "pool_weights": [float(x) for x in W],
                           "test_time_counterpart": "token_pooled_AVERAGED jacobian (J-lens vector)",
                           "norm": "none"},
            "tokens": {"injection_char": ic, "injection_token_id": iid,
                       "injection_left_neighbor_id": left,
                       "injection_right_neighbor_id": right, "critic_suffix_ids": None},
            "prompt_templates": {"actor": SINGLE_TEMPLATE},
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "created_by": "pretrain.build_pooled_single"}
    for o in (args.out_train, args.out_val):
        with open(o + ".nla_meta.yaml", "w") as fh:
            yaml.safe_dump(side, fh, sort_keys=False, allow_unicode=True)
    json.dump(n, open(args.out_train + ".meta.json", "w"), indent=2)
    print(json.dumps(n), flush=True)


if __name__ == "__main__":
    main()
