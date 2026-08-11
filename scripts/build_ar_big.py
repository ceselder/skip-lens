#!/usr/bin/env python3
"""Transform fl_big AV data (485k rows) into AR-SFT format, streaming.

Same transform as scripts/build_ar_data.py on the 50,925-row set:
  prompt = critic template filled with stripped AV response; activation_vector unchanged.
Extras: streams row-group by row-group (writer), drops exact-prompt collisions with the
ar_ablation heldout set, verifies critic_suffix_ids on a sample of prompts.
"""
import os, random, sys, yaml
from datetime import datetime, timezone
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "/workspace/cnla/skip-lens")

TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
# Paths are env-overridable so this reformats any AV parquet (real-span, fl_big, ...).
SRC = os.environ.get("AR_SRC", "/workspace/cnla/skip-lens/data/fl_big/sft_train.parquet")
OUT = Path(os.environ.get("AR_OUT", "/workspace/cnla/data/ar_L62_big/train.parquet"))
OUT_DIR = OUT.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)
REF_SIDECAR = os.environ.get("AR_REF_SIDECAR", "/workspace/cnla/data/sidecar_ar_L62_train.parquet.nla_meta.yaml")
HELDOUT = os.environ.get("AR_HELDOUT", "/workspace/cnla/data/ar_ablation/heldout.parquet")

heldout_prompts = set(pq.read_table(HELDOUT, columns=["prompt"]).column("prompt").to_pylist())
print(f"[ar-big] heldout prompts loaded: {len(heldout_prompts)}", flush=True)

pf = pq.ParquetFile(SRC)
schema = pa.schema([
    ("prompt", pa.string()),
    ("activation_vector", pa.list_(pa.float32())),
    ("doc_id", pa.string()),
])
writer = pq.ParquetWriter(OUT, schema)
n_total = n_written = n_empty = n_contam = 0
sample_prompts = []
rng = random.Random(0)
for rg in range(pf.num_row_groups):
    t = pf.read_row_group(rg, columns=["response", "activation_vector", "doc_id"])
    resp = t.column("response").to_pylist()
    keep_idx, prompts = [], []
    for i, r in enumerate(resp):
        e = (r or "").strip()
        if not e:
            n_empty += 1
            continue
        p = TEMPLATE.format(explanation=e)
        if p in heldout_prompts:
            n_contam += 1
            continue
        keep_idx.append(i)
        prompts.append(p)
    n_total += len(resp)
    sub = t.take(keep_idx)
    chunk = pa.table({
        "prompt": pa.array(prompts, type=pa.string()),
        "activation_vector": sub.column("activation_vector").cast(pa.list_(pa.float32())),
        "doc_id": sub.column("doc_id").cast(pa.string()),
    }).cast(schema)
    writer.write_table(chunk)
    n_written += len(prompts)
    for p in prompts:
        if len(sample_prompts) < 200 and rng.random() < 0.002:
            sample_prompts.append(p)
    if rg % 40 == 0:
        print(f"[ar-big] rg {rg}/{pf.num_row_groups} written={n_written}", flush=True)
writer.close()
print(f"[ar-big] DONE total={n_total} written={n_written} empty={n_empty} heldout_collisions={n_contam}", flush=True)

# --- verify: suffix ids + act length ---
from transformers import AutoTokenizer
from nla.datagen.injection_tokens import compute_critic_suffix_ids
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
suffix_ids = compute_critic_suffix_ids(tok, TEMPLATE)
print(f"[ar-big] critic_suffix_ids = {suffix_ids}", flush=True)
with open(REF_SIDECAR) as f:
    ref = yaml.safe_load(f)
assert suffix_ids == ref["tokens"]["critic_suffix_ids"], (suffix_ids, ref["tokens"]["critic_suffix_ids"])

pfo = pq.ParquetFile(OUT)
first = pfo.read_row_group(0).to_pylist()
check = [first[0]["prompt"]] + sample_prompts
bad = 0
for p in check:
    ids = tok.encode(p, add_special_tokens=False)
    if ids[-len(suffix_ids):] != suffix_ids:
        bad += 1
        print(f"[ar-big] SUFFIX MISMATCH: ...{ids[-len(suffix_ids):]} prompt tail={p[-60:]!r}", flush=True)
assert bad == 0, f"{bad}/{len(check)} prompts failed suffix check"
print(f"[ar-big] suffix check OK on {len(check)} prompts; act len row0 = {len(first[0][activation_vector])}", flush=True)
assert len(first[0]["activation_vector"]) == 5120
assert pfo.metadata.num_rows == n_written

# --- sidecar ---
ref["stage"] = "ar_sft"
ref["dataset_id"] = "ar_naive_fl_big_Qwen3.6-27B_L62"
ref["row_count"] = n_written
ref["tokens"]["critic_suffix_ids"] = suffix_ids
ref.setdefault("prompt_templates", {})["ar"] = TEMPLATE
ref["created_by"] = "scripts.build_ar_big"
ref["created_at"] = datetime.now(timezone.utc).isoformat()
ref["source"] = SRC
with open(str(OUT) + ".nla_meta.yaml", "w") as f:
    yaml.safe_dump(ref, f, sort_keys=False, allow_unicode=True)
print(f"[ar-big] sidecar written -> {OUT}.nla_meta.yaml (row_count={n_written})", flush=True)
