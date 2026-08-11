#!/usr/bin/env python3
"""Truncate stored fed-layer readouts to the first K tokens — a valid short sample that drops the
forced-length (min=max=24) filler. Re-tokenizes each readout string, keeps first K, re-decodes.
Args: K file1 file2 ...  -> writes <file>_{K}tok.json (jlens_top / actual unchanged)."""
import sys, json
from transformers import AutoTokenizer

K = int(sys.argv[1])
files = sys.argv[2:]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
for rf in files:
    recs = json.load(open(rf))
    for r in recs:
        r["readout"] = [tok.decode(tok.encode(ro, add_special_tokens=False)[:K]).strip()
                        for ro in r.get("readout", [])]
    out = rf.replace(".json", f"_{K}tok.json")
    json.dump(recs, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {out} ({len(recs)} recs, readout -> {K} tok)", flush=True)
