"""GPU evaluation for OPD/SFT future lenses.

Reports teacher/student KL, teacher top-1 agreement, EOS behavior, continuation
length, and token-exact reference agreement.  It also writes decoded examples
for the separate Sonnet-5 coherence/hallucination judge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.opd import teacher_student_kl
from nla.train_opd import _predictive_logits, _right_pad, _student_prompt_ids, _trim_generated
from nla.utils import register_karvonen_hook


def load_rows(path, feed_col, limit):
    pf = pq.ParquetFile(path)
    cols = ["prompt", "activation_vector", "teacher_input_ids", "target_ids", "ctx_text"]
    if feed_col != "activation_vector":
        cols.append(feed_col)
    rows = []
    for rg_idx in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_idx, columns=cols)
        for row in rg.to_pylist():
            row["feed"] = row[feed_col]
            rows.append(row)
            if limit and len(rows) >= limit:
                return rows
    return rows


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--av-ckpt", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--feed-col", default="activation_vector")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--max-teacher-context", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.sidecar = args.sidecar or args.parquet
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = load_nla_config(args.sidecar, tok)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    model = PeftModel.from_pretrained(base, args.av_ckpt).eval()
    vref = [None]
    register_karvonen_hook(
        model, vref, cfg.injection_token_id,
        cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id)
    rows = load_rows(args.parquet, args.feed_col, args.max_rows)
    eos_ids = {tok.eos_token_id}
    geos = getattr(model.generation_config, "eos_token_id", None)
    eos_ids.update([geos] if isinstance(geos, int) else (geos or []))
    detail = []
    per_horizon = {i: {"kl": [], "top1": [], "teacher_lp": []}
                   for i in range(args.max_new_tokens)}

    for b0 in range(0, len(rows), args.batch_size):
        batch = rows[b0:b0 + args.batch_size]
        prompt = _student_prompt_ids(batch, tok, cfg.injection_char)
        pids = torch.tensor([prompt], dtype=torch.long, device=device).repeat(len(batch), 1)
        feeds = torch.tensor(np.stack([r["feed"] for r in batch]),
                             dtype=torch.float32, device=device)
        vref[0] = feeds
        try:
            with torch.no_grad():
                gen = model.generate(
                    input_ids=pids, attention_mask=torch.ones_like(pids),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    **({"temperature": args.temperature, "top_p": 1.0, "top_k": 0}
                       if args.temperature > 0 else {}),
                    pad_token_id=tok.eos_token_id)
        finally:
            vref[0] = None
        responses = _trim_generated(gen, len(prompt), eos_ids)
        width = max(len(x) for x in responses)
        valid = torch.zeros((len(batch), width), dtype=torch.bool, device=device)
        for i, response in enumerate(responses):
            valid[i, :len(response)] = True

        contexts = [list(map(int, r["teacher_input_ids"][-args.max_teacher_context:]))
                    for r in batch]
        tseqs = [c + y for c, y in zip(contexts, responses)]
        tids, tattn, _ = _right_pad(tseqs, tok.pad_token_id, device)
        tplens = torch.tensor([len(c) for c in contexts], device=device)
        with torch.no_grad(), model.disable_adapter():
            tl = model(input_ids=tids, attention_mask=tattn, use_cache=False).logits
            tp = _predictive_logits(tl, tplens, width).float()
        sseqs = [prompt + y for y in responses]
        sids, sattn, _ = _right_pad(sseqs, tok.pad_token_id, device)
        splens = torch.full((len(batch),), len(prompt), device=device, dtype=torch.long)
        vref[0] = feeds
        try:
            with torch.no_grad():
                sl = model(input_ids=sids, attention_mask=sattn, use_cache=False).logits
                sp = _predictive_logits(sl, splens, width).float()
        finally:
            vref[0] = None
        kl = teacher_student_kl(tp, sp)
        top1 = tp.argmax(-1) == sp.argmax(-1)
        tlogp = F.log_softmax(tp, -1)

        for i, (row, response) in enumerate(zip(batch, responses)):
            teacher_lp = [float(tlogp[i, j, response[j]]) for j in range(len(response))]
            ref = list(map(int, row["target_ids"][:args.max_new_tokens]))
            prefix = 0
            for a, b in zip(response, ref):
                if a != b:
                    break
                prefix += 1
            rec = {
                "index": b0 + i, "feed_col": args.feed_col,
                "context": row.get("ctx_text", ""),
                "readout": tok.decode(response, skip_special_tokens=True).strip(),
                "reference": tok.decode(ref, skip_special_tokens=True).strip(),
                "response_ids": response, "reference_ids": ref,
                "length": len(response), "ended_eos": bool(response and response[-1] in eos_ids),
                "exact_prefix_tokens": prefix,
                "mean_kl": float(kl[i, :len(response)].mean()),
                "kl_by_token": [float(x) for x in kl[i, :len(response)]],
                "top1_agreement": float(top1[i, :len(response)].float().mean()),
                "mean_teacher_logprob": mean(teacher_lp),
            }
            detail.append(rec)
            for j in range(len(response)):
                per_horizon[j]["kl"].append(float(kl[i, j]))
                per_horizon[j]["top1"].append(float(top1[i, j]))
                per_horizon[j]["teacher_lp"].append(teacher_lp[j])
        print(f"[eval] {min(b0 + len(batch), len(rows))}/{len(rows)}", flush=True)

    aggregate = {
        "n": len(detail), "feed_col": args.feed_col,
        "mean_kl": mean([x["mean_kl"] for x in detail]),
        "top1_agreement": mean([x["top1_agreement"] for x in detail]),
        "mean_teacher_logprob": mean([x["mean_teacher_logprob"] for x in detail]),
        "mean_length": mean([x["length"] for x in detail]),
        "eos_rate": mean([x["ended_eos"] for x in detail]),
        "mean_exact_prefix_tokens": mean([x["exact_prefix_tokens"] for x in detail]),
        "full_reference_match": mean([
            x["response_ids"][:len(x["reference_ids"])] == x["reference_ids"]
            for x in detail]),
    }
    first_token_kl = [x["kl_by_token"][0] for x in detail if x["kl_by_token"]]
    aggregate["first_token_kl_quantiles"] = {
        str(q): float(np.quantile(first_token_kl, q))
        for q in (0.5, 0.9, 0.95, 0.99)
    }
    horizon = {
        str(i + 1): {"n": len(v["kl"]), "mean_kl": mean(v["kl"]),
                     "top1_agreement": mean(v["top1"]),
                     "mean_teacher_logprob": mean(v["teacher_lp"])}
        for i, v in per_horizon.items() if v["kl"]
    }
    out = {"checkpoint": args.av_ckpt, "aggregate": aggregate,
           "by_horizon": horizon, "detail": detail}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
