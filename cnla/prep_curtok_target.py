#!/usr/bin/env python3
"""Current-token-inclusive target for a MISMATCHED skip-lens: same 150k L62-source corpus as
skiplens_L62_150k, but the reconstruction target = [current token] + [future continuation]
(the current token = the token at the activation's own position = last token of ctx_text).
Activation (L62) unchanged; only `response` changes. Directly comparable to the future-only arm."""
import sys, os, shutil
import pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoTokenizer

N = int(sys.argv[1]) if len(sys.argv) > 1 else 150000
SRC = f"/workspace/cnla/skip-lens/data/fl_big/av_L62_{N // 1000}k.parquet"
OUT = f"/workspace/cnla/skip-lens/data/fl_big/av_L62_curtok_{N // 1000}k.parquet"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")

tbl = pq.read_table(SRC)
ctx = tbl.column("ctx_text").to_pylist()
resp = tbl.column("response").to_pylist()
new_resp = []
for c, r in zip(ctx, resp):
    ids = tok.encode(c if (c and c.strip()) else ".", add_special_tokens=False)
    cur = tok.decode([ids[-1]]) if ids else ""
    new_resp.append(cur + (r or ""))

cols = {n: tbl.column(n) for n in tbl.schema.names}
cols["response"] = pa.array(new_resp, type=pa.string())
pq.write_table(pa.table(cols), OUT, row_group_size=2000)
sc = SRC + ".nla_meta.yaml"
if os.path.exists(sc):
    shutil.copy(sc, OUT + ".nla_meta.yaml")

for i in range(4):
    print(f"[{i}] CTX tail : ...{ctx[i][-45:]!r}")
    print(f"    current  : {tok.decode([tok.encode(ctx[i], add_special_tokens=False)[-1]])!r}")
    print(f"    OLD resp : {resp[i][:55]!r}")
    print(f"    NEW resp : {new_resp[i][:60]!r}")
print(f"wrote {OUT} ({len(new_resp)} rows, response = current-token + future)", flush=True)
