"""Build the skip-lens FEED parquet: av_train.parquet + a NEW `act_L42` column.

Unlike extract_L42.py (which REPLACES activation_vector with L42), this keeps
activation_vector = L62 UNCHANGED (the reward/eval GOLD) and ADDS act_L42 (the
vector injected into block 1 when training with `--feed-col act_L42`). Every
row stays ALIGNED to av_train.parquet — nothing is dropped, filtered, or
reordered; an empty ctx_text forwards "." as a placeholder (same as
extract_L42.py). Forwards run with adapters off (bare base model = clean
residuals), grabbing the block-42 residual at the LAST REAL token (right-pad,
attention_mask.sum(1)-1).

Run on a GPU box; respects CUDA_VISIBLE_DEVICES from the environment.
"""
import os
import shutil
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.utils.arch_adapters import resolve_text_model

# SRC / OUT / LAYER overridable via argv (SRC OUT [LAYER]) so the same builder
# serves the fresh disjoint corpus (rl_fresh) as well as av_train.
SRC = sys.argv[1] if len(sys.argv) > 1 else "data/cnla/av_train.parquet"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/cnla/av_train_feedL42.parquet"
BASE = "Qwen/Qwen3.6-27B"
LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 42
FEED_COL = f"act_L{LAYER}"
dev = "cuda"

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
layers = resolve_text_model(model).model.layers
grab = {}
layers[LAYER].register_forward_hook(
    lambda m, i, o: grab.__setitem__(LAYER, (o[0] if isinstance(o, tuple) else o).detach()))
torch.set_grad_enabled(False)

tbl = pq.read_table(SRC)
assert FEED_COL not in tbl.schema.names, f"{SRC} already has a {FEED_COL} column"
ctx = tbl.column("ctx_text").to_pylist()
acts42, B = [], 16
tok.padding_side = "right"
for c0 in range(0, len(ctx), B):
    chunk = [t if (t and t.strip()) else "." for t in ctx[c0:c0 + B]]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
              max_length=512).to(dev)
    model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    lastreal = enc.attention_mask.sum(1) - 1        # last real token per row (right-pad)
    for i in range(len(chunk)):
        acts42.append(grab[LAYER][i, int(lastreal[i])]
                      .float().cpu().numpy().astype("float32").tolist())
    if c0 % 800 == 0:
        print(f"L{LAYER} extract {c0}/{len(ctx)}", flush=True)

assert len(acts42) == tbl.num_rows, (
    f"row misalignment: extracted {len(acts42)} vectors vs {tbl.num_rows} source rows")
# ALL original columns unchanged (activation_vector stays L62) + the new feed col.
cols = {n: tbl.column(n) for n in tbl.schema.names}
cols[FEED_COL] = pa.array(acts42, type=pa.list_(pa.float32()))
pq.write_table(pa.table(cols), OUT, row_group_size=2000)
# copy the sidecar meta (same inj/mse_scale/d_model; gold column is unchanged)
for suf in (".nla_meta.yaml",):
    if os.path.exists(SRC + suf):
        shutil.copy(SRC + suf, OUT + suf)
print(f"wrote {OUT} ({len(acts42)} rows, +{FEED_COL}; activation_vector stays L62)")
