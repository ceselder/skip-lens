"""Train an activation-only lens with on-policy distribution distillation.

The frozen teacher and trainable student share one base-model allocation:

* teacher forward: adapter disabled, real source text prefilled;
* student rollout/forward: adapter enabled, only the activation is injected;
* both predict along the student's sampled prefix;
* loss: KL(teacher || student), until the first KL threshold violation;
* violation: supervise EOS once, then mask the remainder of the rollout.

``--objective sft`` provides the matched continuation-SFT control using the
same loader, actor, optimizer, token counter and wall-clock stopping machinery.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.opd import opd_loss
from nla.train_sft import build_lr_lambda
from nla.utils import build_prompt_text, register_karvonen_hook


def load_dataset(path: str, max_rows: int | None = None) -> list[dict]:
    pf = pq.ParquetFile(path)
    needed = ["prompt", "activation_vector", "teacher_input_ids", "target_ids"]
    missing = set(needed) - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f"{path} missing OPD columns: {sorted(missing)}")
    rows: list[dict] = []
    for rg_idx in range(pf.num_row_groups):
        if max_rows is not None and len(rows) >= max_rows:
            break
        rg = pf.read_row_group(rg_idx, columns=needed)
        acts_col = rg.column("activation_vector").combine_chunks()
        acts = np.asarray(acts_col.flatten(), dtype=np.float32).reshape(len(acts_col), -1)
        prompts = rg.column("prompt").to_pylist()
        contexts = rg.column("teacher_input_ids").to_pylist()
        targets = rg.column("target_ids").to_pylist()
        take = len(prompts) if max_rows is None else min(len(prompts), max_rows - len(rows))
        for i in range(take):
            if contexts[i] and targets[i]:
                rows.append({
                    "prompt": prompts[i],
                    "activation": acts[i],
                    "teacher_input_ids": contexts[i],
                    "target_ids": targets[i],
                })
    if not rows:
        raise ValueError(f"no usable rows in {path}")
    return rows


def _right_pad(seqs: list[list[int]], pad_id: int, device: str):
    lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long, device=device)
    width = int(lengths.max().item())
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, : len(seq)] = 1
    return ids, mask, lengths


def _trim_generated(sequences: torch.Tensor, prompt_len: int, eos_ids: set[int]):
    out: list[list[int]] = []
    for row in sequences[:, prompt_len:].tolist():
        end = next((i + 1 for i, tok in enumerate(row) if tok in eos_ids), len(row))
        out.append(row[:end])
    return out


def _predictive_logits(logits: torch.Tensor, prefix_lens: torch.Tensor, width: int):
    """Gather distributions predicting response tokens 0..width-1."""
    offsets = torch.arange(width, device=logits.device).view(1, -1)
    positions = prefix_lens.to(logits.device).view(-1, 1) - 1 + offsets
    # Some rows have shorter responses than ``width``. Their extra positions
    # are masked later, but gathering must still stay in bounds.
    positions = positions.clamp(max=logits.shape[1] - 1)
    rows = torch.arange(logits.shape[0], device=logits.device).view(-1, 1)
    return logits[rows, positions]


def _student_prompt_ids(rows, tokenizer, injection_char):
    texts = [build_prompt_text(r["prompt"], injection_char, tokenizer) for r in rows]
    ids = [tokenizer.encode(x, add_special_tokens=False) for x in texts]
    if len({tuple(x) for x in ids}) != 1:
        raise ValueError("OPD currently requires one canonical actor prompt per batch")
    return ids[0]


def _save(model, tokenizer, sidecar: str, save_dir: Path, step: int, meta: dict):
    out = save_dir / f"iter_{step:07d}"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    src = Path(sidecar)
    if src.suffix == ".parquet":
        src = Path(str(src) + ".nla_meta.yaml")
    if src.exists():
        shutil.copy2(src, out / "nla_meta.yaml")
    (out / "opd_meta.json").write_text(json.dumps(meta, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["opd", "sft"], required=True)
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--av-ckpt", required=True, help="short future-lens SFT LoRA")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--sidecar", default=None)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--num-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--max-teacher-context", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--kl-threshold", type=float, default=None)
    ap.add_argument("--eos-weight", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--min-lr", type=float, default=3e-6)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-optimized-tokens", type=int, default=0,
                    help="0 disables; otherwise stop after this many loss-bearing tokens")
    ap.add_argument("--max-wall-seconds", type=float, default=0.0,
                    help="0 disables; used for the GPU-hour-matched SFT arm")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--wandb-project", default="skip-lens-opd")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-group", default="opd-vs-sft")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    args.sidecar = args.sidecar or args.parquet

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = load_nla_config(args.sidecar, tokenizer)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model = PeftModel.from_pretrained(base, args.av_ckpt, is_trainable=True)
    vectors_ref = [None]
    register_karvonen_hook(
        model, vectors_ref, cfg.injection_token_id,
        cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id,
    )
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("AV checkpoint loaded with no trainable adapter parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        build_lr_lambda(args.warmup_steps, args.num_steps, args.min_lr / args.lr),
    )

    rows = load_dataset(args.parquet, args.max_rows)
    rng = np.random.default_rng(args.seed)
    order = np.arange(len(rows))
    rng.shuffle(order)
    cursor = 0
    eos_ids = {tokenizer.eos_token_id}
    generation_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(generation_eos, int):
        eos_ids.add(generation_eos)
    elif generation_eos:
        eos_ids.update(generation_eos)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "resolved_config.json").write_text(json.dumps(vars(args), indent=2))
    history_path = save_dir / "metrics.jsonl"
    wb = None
    if not args.no_wandb:
        import wandb
        wb = wandb.init(project=args.wandb_project, name=args.wandb_name,
                        group=args.wandb_group, config=vars(args))

    optimized_total = 0
    wall_start = time.monotonic()
    last_step = 0
    last_meta: dict = {}
    for step in range(1, args.num_steps + 1):
        if cursor + args.batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        batch = [rows[int(i)] for i in order[cursor:cursor + args.batch_size]]
        cursor += args.batch_size
        prompt = _student_prompt_ids(batch, tokenizer, cfg.injection_char)
        prompt_len = len(prompt)
        activations = torch.tensor(
            np.stack([r["activation"] for r in batch]), dtype=torch.float32, device=device)
        t0 = time.monotonic()

        optimizer.zero_grad(set_to_none=True)
        if args.objective == "opd":
            # Roll out the current student. No gradients are retained through sampling.
            pids = torch.tensor([prompt], dtype=torch.long, device=device).repeat(len(batch), 1)
            vectors_ref[0] = activations
            model.eval()
            try:
                with torch.no_grad():
                    generated = model.generate(
                        input_ids=pids, attention_mask=torch.ones_like(pids),
                        max_new_tokens=args.max_new_tokens, do_sample=True,
                        temperature=args.temperature, top_p=1.0, top_k=0,
                        repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id,
                    )
            finally:
                vectors_ref[0] = None
            responses = _trim_generated(generated, prompt_len, eos_ids)
        else:
            responses = [list(map(int, r["target_ids"][:args.max_new_tokens])) for r in batch]

        width = max(len(x) for x in responses)
        valid = torch.zeros((len(batch), width), dtype=torch.bool, device=device)
        for i, response in enumerate(responses):
            valid[i, : len(response)] = True

        student_seqs = [prompt + response for response in responses]
        sids, sattn, splens = _right_pad(student_seqs, tokenizer.pad_token_id, device)
        # Every student sequence has the same prompt prefix.
        student_prefix_lens = torch.full_like(splens, prompt_len)

        teacher_pred = None
        if args.objective == "opd":
            teacher_contexts = [
                list(map(int, r["teacher_input_ids"][-args.max_teacher_context:]))
                for r in batch
            ]
            teacher_seqs = [c + response for c, response in zip(teacher_contexts, responses)]
            tids, tattn, _ = _right_pad(teacher_seqs, tokenizer.pad_token_id, device)
            teacher_prefix_lens = torch.tensor(
                [len(x) for x in teacher_contexts], dtype=torch.long, device=device)
            model.eval()
            with torch.no_grad(), model.disable_adapter():
                vectors_ref[0] = None
                tout = model(input_ids=tids, attention_mask=tattn, use_cache=False).logits
                teacher_pred = _predictive_logits(tout, teacher_prefix_lens, width).float()
            del tout, tids, tattn

        model.train()
        vectors_ref[0] = activations
        try:
            sout = model(input_ids=sids, attention_mask=sattn, use_cache=False).logits
            student_pred = _predictive_logits(sout, student_prefix_lens, width)
            if args.objective == "opd":
                loss_out = opd_loss(
                    teacher_pred, student_pred, valid, tokenizer.eos_token_id,
                    args.kl_threshold, args.eos_weight,
                )
                loss = loss_out.loss
                optimized = loss_out.optimized_tokens
                with torch.no_grad():
                    agree = ((teacher_pred.argmax(-1) == student_pred.argmax(-1)) & valid)
                    mean_kl = float(loss_out.kl[valid].mean())
                    cutoff_rate = float(loss_out.eos_mask.any(-1).float().mean())
                    horizon = float((loss_out.distill_mask | loss_out.eos_mask).sum(-1).float().mean())
                metrics = {
                    "distill_loss": float(loss_out.distill_loss.detach()),
                    "eos_loss": float(loss_out.eos_loss.detach()),
                    "teacher_student_kl": mean_kl,
                    "top1_agreement": float(agree.sum() / valid.sum().clamp_min(1)),
                    "cutoff_rate": cutoff_rate,
                    "effective_horizon": horizon,
                }
            else:
                targets = torch.full((len(batch), width), -100, dtype=torch.long, device=device)
                for i, response in enumerate(responses):
                    targets[i, : len(response)] = torch.tensor(response, device=device)
                loss = F.cross_entropy(
                    student_pred.float().reshape(-1, student_pred.shape[-1]),
                    targets.reshape(-1), ignore_index=-100,
                )
                optimized = int(valid.sum().item())
                metrics = {"sft_ppl": math.exp(min(20.0, float(loss.detach())))}
            loss.backward()
        finally:
            vectors_ref[0] = None

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimized_total += optimized
        elapsed = time.monotonic() - wall_start
        last_step = step
        log = {
            "step": step, "objective": args.objective, "loss": float(loss.detach()),
            "optimized_tokens": optimized, "optimized_tokens_total": optimized_total,
            "sampled_tokens": int(valid.sum().item()),
            "lr": scheduler.get_last_lr()[0], "grad_norm": float(grad_norm),
            "step_seconds": time.monotonic() - t0, "wall_seconds": elapsed,
            **metrics,
        }
        with history_path.open("a") as f:
            f.write(json.dumps(log) + "\n")
        if wb is not None:
            wb.log(log, step=step)
        print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in log.items() if k not in {"objective"}), flush=True)

        last_meta = {
            "objective": args.objective, "steps": step,
            "optimized_tokens": optimized_total, "wall_seconds": elapsed,
            "kl_threshold": args.kl_threshold, "eos_weight": args.eos_weight,
        }
        should_stop = (
            (args.max_optimized_tokens > 0 and optimized_total >= args.max_optimized_tokens)
            or (args.max_wall_seconds > 0 and elapsed >= args.max_wall_seconds)
        )
        if step % args.save_every == 0 or should_stop or step == args.num_steps:
            out = _save(model, tokenizer, args.sidecar, save_dir, step, last_meta)
            print(f"[save] {out}", flush=True)
        if should_stop:
            print(f"[stop] token/wall budget reached at step {step}", flush=True)
            break

    (save_dir / "run_summary.json").write_text(json.dumps(last_meta, indent=2))
    if wb is not None:
        wb.finish()
    print(f"done: steps={last_step} optimized_tokens={optimized_total}", flush=True)


if __name__ == "__main__":
    main()
