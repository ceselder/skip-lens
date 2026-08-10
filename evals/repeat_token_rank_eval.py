"""Exact target-token ranks for the scaled repeat-after-me Skip-Lens control.

Unlike ``opd_eval``, this rank-only evaluator skips generation and the full-context
teacher forward.  It teacher-forces the held-out eight-token reference through the
activation-only student and records the target token's full-vocabulary rank at each
position for every requested activation column.
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
from nla.train_opd import _predictive_logits, _right_pad, _student_prompt_ids
from nla.utils import register_karvonen_hook


def exact_ranks(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return one-indexed target ranks; ties receive their best possible rank."""
    target_logits = logits.gather(-1, targets.unsqueeze(-1))
    return (logits > target_logits).sum(-1) + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--av-ckpt", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--feed-cols", default="activation_vector,act_L42")
    ap.add_argument("--max-rows", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.sidecar = args.sidecar or args.parquet
    feed_cols = [x.strip() for x in args.feed_cols.split(",") if x.strip()]

    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = load_nla_config(args.sidecar, tok)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    model = PeftModel.from_pretrained(base, args.av_ckpt).eval()
    vref = [None]
    register_karvonen_hook(
        model, vref, cfg.injection_token_id,
        cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id,
    )

    cols = ["prompt", "target_ids", *feed_cols]
    table = pq.read_table(args.parquet, columns=cols).slice(0, args.max_rows)
    rows = table.to_pylist()
    prompt = _student_prompt_ids(rows[:1], tok, cfg.injection_char)
    by_feed = {col: [[] for _ in range(args.max_tokens)] for col in feed_cols}
    nll_by_feed = {col: [[] for _ in range(args.max_tokens)] for col in feed_cols}

    for b0 in range(0, len(rows), args.batch_size):
        batch = rows[b0:b0 + args.batch_size]
        refs = [list(map(int, r["target_ids"][:args.max_tokens])) for r in batch]
        width = max(map(len, refs))
        ids, attn, _ = _right_pad([prompt + ref for ref in refs], tok.pad_token_id, "cuda")
        plens = torch.full((len(batch),), len(prompt), device="cuda", dtype=torch.long)
        targets = torch.full((len(batch), width), tok.pad_token_id,
                             device="cuda", dtype=torch.long)
        valid = torch.zeros((len(batch), width), device="cuda", dtype=torch.bool)
        for i, ref in enumerate(refs):
            targets[i, :len(ref)] = torch.tensor(ref, device="cuda")
            valid[i, :len(ref)] = True

        for col in feed_cols:
            vref[0] = torch.tensor(np.stack([r[col] for r in batch]),
                                   dtype=torch.float32, device="cuda")
            try:
                with torch.no_grad():
                    logits = model(input_ids=ids, attention_mask=attn,
                                   use_cache=False).logits
                    pred = _predictive_logits(logits, plens, width).float()
            finally:
                vref[0] = None
            ranks = exact_ranks(pred, targets)
            nll = -F.log_softmax(pred, -1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            for i, ref in enumerate(refs):
                for j in range(len(ref)):
                    by_feed[col][j].append(int(ranks[i, j]))
                    nll_by_feed[col][j].append(float(nll[i, j]))
            del logits, pred, ranks, nll
        print(f"[rank] {min(b0 + len(batch), len(rows))}/{len(rows)}", flush=True)

    result = {
        "checkpoint": args.av_ckpt,
        "n": len(rows),
        "vocab_size": len(tok),
        "max_tokens": args.max_tokens,
        "feed_cols": {},
    }
    for col in feed_cols:
        result["feed_cols"][col] = {
            "rank_by_position": by_feed[col],
            "nll_by_position": nll_by_feed[col],
            "summary_by_position": [
                {
                    "position": j + 1,
                    "n": len(rs),
                    "top1_accuracy": float(np.mean(np.asarray(rs) == 1)),
                    "median_rank": float(np.median(rs)),
                    "mean_log10_rank": float(np.mean(np.log10(rs))),
                    "rank_quantiles": {
                        str(q): float(np.quantile(rs, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
                    },
                }
                for j, rs in enumerate(by_feed[col])
            ],
        }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
