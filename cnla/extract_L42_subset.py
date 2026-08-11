#!/usr/bin/env python3
"""L42 twin of the L62 subset: forward each ctx_text (clean base residuals), grab the block-42
residual at the last real token, and write av_L42_{N}k.parquet with activation_vector := L42
(everything else identical to the L62 subset). Position-matched to how fl_big's L62 was collected
(last ctx token). Same recipe as cnla/extract_L42.py, parameterized for the fl_big subset."""
import sys, os, shutil
import pyarrow as pa, pyarrow.parquet as pq, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.utils.arch_adapters import resolve_text_model

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
SRC = f"/workspace/cnla/skip-lens/data/fl_big/av_L62_{N // 1000}k.parquet"
OUT = f"/workspace/cnla/skip-lens/data/fl_big/av_L42_{N // 1000}k.parquet"
BASE = "Qwen/Qwen3.6-27B"
dev = "cuda"

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
layers = resolve_text_model(model).model.layers
grab = {}
layers[42].register_forward_hook(lambda m, i, o: grab.__setitem__(42, (o[0] if isinstance(o, tuple) else o).detach()))
torch.set_grad_enabled(False)

tbl = pq.read_table(SRC)
ctx = tbl.column("ctx_text").to_pylist()
acts, B = [], 16
for c0 in range(0, len(ctx), B):
    chunk = [t if (t and t.strip()) else "." for t in ctx[c0:c0 + B]]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
    model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    lastreal = enc.attention_mask.sum(1) - 1                    # last real token per row (right-pad)
    for i in range(len(chunk)):
        acts.append(grab[42][i, int(lastreal[i])].float().cpu().numpy().astype("float32").tolist())
    if c0 % 3200 == 0:
        print(f"L42 extract {c0}/{len(ctx)}", flush=True)

cols = {n: tbl.column(n) for n in tbl.schema.names}
cols["activation_vector"] = pa.array(acts, type=pa.list_(pa.float32()))     # replace L62 -> L42
pq.write_table(pa.table(cols), OUT, row_group_size=2000)
sc = SRC + ".nla_meta.yaml"
if os.path.exists(sc):
    shutil.copy(sc, OUT + ".nla_meta.yaml")
print(f"wrote {OUT} ({len(acts)} rows, activation_vector=L42)", flush=True)
