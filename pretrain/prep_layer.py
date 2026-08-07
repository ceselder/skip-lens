"""Build a per-training-layer AO train parquet from the token-matched multi-layer
labeled data: set activation_vector = act_L{layer}, keep the (layer-independent)
label + metadata, drop the other layers' activation columns. Row order preserved
so a fixed-seed train/val split is identical across layers (token-matched)."""
import argparse, os, shutil
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--labeled", required=True)
ap.add_argument("--layer", type=int, required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

t = pq.read_table(args.labeled)
keep = ["prompt", "rollouts", "top_tokens", "ctx_text", "description", "doc_id", "next_token_entropy"]
cols = {n: t.column(n) for n in keep if n in t.column_names}
cols["activation_vector"] = t.column(f"act_L{args.layer}")   # this layer's token-matched activation
out = pa.table(cols)
pq.write_table(out, args.out, row_group_size=2000)
meta = args.labeled + ".meta.json"
if os.path.exists(meta):
    shutil.copy(meta, args.out + ".meta.json")
print(f"wrote {args.out}: {out.num_rows} rows, activation_vector <- act_L{args.layer}")
