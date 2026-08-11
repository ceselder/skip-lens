"""Re-extract the block-42 (middle) residual for every RL example, so we can RL the L62-trained
AVs while FEEDING the middle layer instead of the penultimate. Forwards each ctx_text (adapters
off = clean base residuals), grabs L42 at the last real token, and writes av_train_L42.parquet
with activation_vector := L42 (everything else identical)."""
import pyarrow.parquet as pq, pyarrow as pa, torch, shutil, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.utils.arch_adapters import resolve_text_model

SRC = "data/cnla/av_train.parquet"
OUT = "data/cnla/av_train_L42.parquet"
BASE = "Qwen/Qwen3.6-27B"
dev = "cuda"
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
layers = resolve_text_model(model).model.layers
grab = {}
layers[42].register_forward_hook(lambda m, i, o: grab.__setitem__(42, (o[0] if isinstance(o, tuple) else o).detach()))
torch.set_grad_enabled(False)

tbl = pq.read_table(SRC)
ctx = tbl.column("ctx_text").to_pylist()
acts42, B = [], 16
tok.padding_side = "right"
for c0 in range(0, len(ctx), B):
    chunk = [t if (t and t.strip()) else "." for t in ctx[c0:c0 + B]]
    enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).to(dev)
    model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
    lastreal = enc.attention_mask.sum(1) - 1                       # last real token per row (right-pad)
    for i in range(len(chunk)):
        acts42.append(grab[42][i, int(lastreal[i])].float().cpu().numpy().astype("float32").tolist())
    if c0 % 800 == 0:
        print(f"L42 extract {c0}/{len(ctx)}", flush=True)

cols = {n: tbl.column(n) for n in tbl.schema.names}
cols["activation_vector"] = pa.array(acts42, type=pa.list_(pa.float32()))   # replace L62 -> L42
pq.write_table(pa.table(cols), OUT, row_group_size=2000)
# reuse the L62 sidecar meta (same inj/mse_scale/d_model; only the fed layer's content differs)
for suf in (".nla_meta.yaml",):
    if os.path.exists(SRC + suf):
        shutil.copy(SRC + suf, OUT + suf)
print(f"wrote {OUT} ({len(acts42)} rows, activation_vector=L42)")
