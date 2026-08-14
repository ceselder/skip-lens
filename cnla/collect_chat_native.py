"""Chat-native training pairs for the futurelens (empty-think anchor).

For each of the first 500 fl_big ctx_texts: wrap as a user turn, apply the chat template with
add_generation_prompt=True + enable_thinking=False (so the prompt ends at the post-<think></think>
assistant anchor), forward to grab the L62 residual AT THAT ANCHOR, and sample the model's on-policy
response from there. Writes av_chat_500.parquet with activation_vector = L62-anchor and
response = on-policy chat response. prompt (ACTOR template) / ctx_text / doc_id copied unchanged so
train_sft's injection is identical; only the SOURCE activation is now chat-native.
"""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.6-27B"
dev = "cuda"
CTX = 256
N = 500
GEN = 24
SRC = "/workspace/cnla/skip-lens/data/fl_big/av_L62_150k.parquet"
OUT = "/workspace/cnla/skip-lens/data/meansub/av_chat_500.parquet"

pf = pq.ParquetFile(SRC)
parts, have = [], 0
for rg in range(pf.num_row_groups):
    parts.append(pf.read_row_group(rg))
    have += parts[-1].num_rows
    if have >= N:
        break
full = pa.concat_tables(parts).slice(0, N)
ctxs = full["ctx_text"].to_pylist()

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)
grab = {}
model.model.layers[62].register_forward_hook(
    lambda m, i, o: grab.__setitem__(62, (o[0] if isinstance(o, tuple) else o).detach()))

acts, resps = [], []
for k, ctx in enumerate(ctxs):
    s = tok.apply_chat_template([{"role": "user", "content": ctx}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    ids = tok(s, return_tensors="pt", add_special_tokens=False).input_ids[:, -CTX:].to(dev)
    model(ids)                                             # populate the L62 grab at the anchor
    acts.append(grab[62][0, -1].float().cpu().numpy().astype("float32").tolist())
    g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=GEN,
                       do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.eos_token_id)
    resps.append(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).strip())
    if k % 50 == 0:
        print(f"[chat-collect] {k}/{N}", flush=True)

ia = full.schema.get_field_index("activation_vector")
ir = full.schema.get_field_index("response")
tb = full.set_column(ia, "activation_vector",
                     pa.array(acts, type=full.schema.field("activation_vector").type))
tb = tb.set_column(ir, "response", pa.array(resps, type=full.schema.field("response").type))
pq.write_table(tb, OUT)
side = yaml.safe_load(open(SRC + ".nla_meta.yaml"))
side["row_count"] = N
yaml.safe_dump(side, open(OUT + ".nla_meta.yaml", "w"), sort_keys=False)
print(f"wrote {OUT} | sample response: {resps[0][:80]!r}")
