"""Self-contained NLA SFT (AV + AR), no Miles dependency.

Single entry point with `--mode {av,ar}`:
  - AV: AutoModelForCausalLM + Karvonen layer-1 injection hook
        loss = cross-entropy on response tokens only
        target: actor learns to verbalise injected activations
  - AR: NLACriticModel (truncated K+1-layer backbone + Linear(d,d) value_head)
        loss = MSE on L2-normalised (pred, gold) at last-token position
        target: critic learns to reconstruct activation from explanation text

Replaces the old Miles-era pipeline (FSDP actor subclass, loss plug-ins,
rollout adapters, shell wrappers, and a separate critic-init script — all
removed in the repo consolidation). AR backbone truncation now happens
in-script.

Loads bf16 model + bitsandbytes AdamW8bit (~4 GB optim states on 8B model
instead of 64 GB for fp32 AdamW). Single GPU; activation memory bounded by
gradient_checkpointing on the AV path.

Saves HF format checkpoints directly — no DCP→HF conversion step.
"""

import argparse
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from nla.utils import critic_predict, register_karvonen_hook
from nla.utils.critic import critic_predict_all


class ResidualMLPHead(torch.nn.Module):
    """Deep AR value head: n pre-norm residual MLP blocks + an identity affine, init ≈ identity.
    Input/output [*, d]. Residual branches are zero-init (fc2 weight/bias = 0) and the final
    affine is the identity, so at init this is EXACTLY the old Linear-identity head — training
    starts from the same reconstruction, then the extra capacity co-trains in. LayerNorm +
    residual stream + GELU = the standard tricks that make deep MLPs trainable."""

    def __init__(self, d, hidden, n_layers=4):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "norm": torch.nn.LayerNorm(d),
                "fc1": torch.nn.Linear(d, hidden),
                "fc2": torch.nn.Linear(hidden, d),
            }) for _ in range(n_layers)
        ])
        self.out = torch.nn.Linear(d, d)
        with torch.no_grad():
            for b in self.blocks:
                torch.nn.init.normal_(b["fc1"].weight, std=0.02); torch.nn.init.zeros_(b["fc1"].bias)
                torch.nn.init.zeros_(b["fc2"].weight); torch.nn.init.zeros_(b["fc2"].bias)  # zero residual branch
            self.out.weight.copy_(torch.eye(d)); torch.nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for b in self.blocks:
            x = x + b["fc2"](torch.nn.functional.gelu(b["fc1"](b["norm"](x))))
        return self.out(x)
from nla.utils.run_config import add_config_arg, apply_config_defaults, save_resolved_config
from nla.config import load_nla_config
from nla.injection import karvonen_inject_in_residual
from nla.models import NLACriticModel
from nla.schema import (
    INJECT_PLACEHOLDER,
    extract_explanation,
    normalize_activation,
    resolve_target_scale,
)


# ----------------------------------------------------------------------------
# Helpers shared with train_rl_self_contained.py (kept inline so this file
# stays self-contained — they're small, and importing creates an awkward
# coupling between SFT and RL trainers).
# ----------------------------------------------------------------------------




def load_sft_dataset(parquet_path, n_max=None, *, mode):
    """Stream-load AV (prompt: list[dict], response: str, activation_vector)
    or AR (prompt: str, activation_vector). Slice rowgroups so n_max=N takes
    only N rows, not the full first rowgroup."""
    cols = (
        ["prompt", "response", "activation_vector"] if mode == "av"
        else ["prompt", "activation_vector"]
    )
    pf = pq.ParquetFile(parquet_path)
    rows = []
    for rg_idx in range(pf.num_row_groups):
        if n_max is not None and len(rows) >= n_max:
            break
        rg = pf.read_row_group(rg_idx, columns=cols)
        n_in_rg = rg.num_rows
        take = n_in_rg if n_max is None else min(n_max - len(rows), n_in_rg)
        rg = rg.slice(0, take)
        # activation_vector via flatten→numpy (zero-copy) — ~100× faster than
        # to_pylist() on 4096-float lists, which builds ~1B PyFloats at 250k rows
        # (GPUs sit idle for 10-20 min otherwise). Same pattern as schema.py.
        acts_col = rg.column("activation_vector").combine_chunks()  # ChunkedArray→Array
        acts_np = (acts_col.flatten().to_numpy(zero_copy_only=False)
                   .astype(np.float32).reshape(len(acts_col), -1))
        prompts = rg.column("prompt").to_pylist()
        responses = rg.column("response").to_pylist() if mode == "av" else None
        for i in range(take):
            row = {"prompt": prompts[i], "activation_vector": acts_np[i]}
            if mode == "av":
                row["response"] = responses[i]
            rows.append(row)
    return rows


def load_heldout_explanation_pairs(parquet_path, n_rows):
    """(explanation, activation) pairs from an AV-split parquet (has `response`).

    The AV split is DOC-DISJOINT from the AR training data by stage-1
    construction, so FVE on these pairs is a genuine held-out number —
    training-batch FVE overstates quality once the data is multi-epoch.
    """
    pf = pq.ParquetFile(parquet_path)
    pairs = []
    for rg_idx in range(pf.num_row_groups):
        if len(pairs) >= n_rows:
            break
        rg = pf.read_row_group(rg_idx, columns=["response", "activation_vector"])
        responses = rg.column("response").to_pylist()
        acts_col = rg.column("activation_vector").combine_chunks()
        acts = (acts_col.flatten().to_numpy(zero_copy_only=False)
                .astype(np.float32).reshape(len(acts_col), -1))
        for resp, act in zip(responses, acts):
            expl = extract_explanation(resp)
            if expl is None:
                continue
            pairs.append((expl, act))
            if len(pairs) >= n_rows:
                break
    return pairs


@torch.no_grad()
def heldout_fve_mse(critic, tokenizer, pairs, template, mse_scale_f, device,
                    micro_batch=16, max_len=1024):
    """Mean per-sample MSE on normalized (pred, gold) over held-out pairs.

    Returns (mean_mse, n_scored). Caller divides by a predict-the-mean
    baseline for FVE. Skips pairs whose critic prompt exceeds max_len
    (would truncate the suffix anchor).
    """
    mses = []
    for cs in range(0, len(pairs), micro_batch):
        chunk = pairs[cs:cs + micro_batch]
        ids_list, golds = [], []
        for expl, act in chunk:
            ids = tokenizer.encode(template.format(explanation=expl),
                                   add_special_tokens=False)
            if not 0 < len(ids) <= max_len:
                continue
            ids_list.append(torch.tensor(ids, dtype=torch.long))
            golds.append(act)
        if not ids_list:
            continue
        bs = len(ids_list)
        T = max(t.numel() for t in ids_list)
        batch_ids = torch.full((bs, T), tokenizer.eos_token_id,
                               dtype=torch.long, device=device)
        attn = torch.zeros((bs, T), dtype=torch.long, device=device)
        for i, t in enumerate(ids_list):
            batch_ids[i, : t.numel()] = t.to(device)
            attn[i, : t.numel()] = 1
        pred = critic_predict(critic, batch_ids, attn, mse_scale_f)
        gold = torch.tensor(np.stack(golds), dtype=torch.float32, device=device)
        pred_n = normalize_activation(pred, mse_scale_f)
        gold_n = normalize_activation(gold, mse_scale_f)
        mses.extend(((pred_n - gold_n) ** 2).mean(dim=-1).tolist())
    return float(np.mean(mses)) if mses else float("nan"), len(mses)


@torch.no_grad()
def heldout_fve_by_dist(critic, tokenizer, pairs, template, mse_scale_f, device,
                        baseline, max_dist=18, micro_batch=16, max_len=1024):
    """FVE reading at each distance-from-last-real-token (d=0 = normal last-token read; larger d
    = truncating the span by d tokens). A dense/all-idx AR stays high across d (truncation-
    resistant); a last-token-only AR peaks at d=0 and decays. Returns [(d, fve%, n)]."""
    sse = np.zeros(max_dist + 1); cnt = np.zeros(max_dist + 1)
    for cs in range(0, len(pairs), micro_batch):
        chunk = pairs[cs:cs + micro_batch]
        ids_list, golds = [], []
        for expl, act in chunk:
            ids = tokenizer.encode(template.format(explanation=expl), add_special_tokens=False)
            if not 0 < len(ids) <= max_len:
                continue
            ids_list.append(torch.tensor(ids, dtype=torch.long)); golds.append(act)
        if not ids_list:
            continue
        bs = len(ids_list); T = max(t.numel() for t in ids_list)
        batch_ids = torch.full((bs, T), tokenizer.eos_token_id, dtype=torch.long, device=device)
        attn = torch.zeros((bs, T), dtype=torch.long, device=device)
        for i, t in enumerate(ids_list):
            batch_ids[i, :t.numel()] = t.to(device); attn[i, :t.numel()] = 1
        pred_all = critic_predict_all(critic, batch_ids, attn, mse_scale_f)  # [B,T,D]
        B_, T_, D_ = pred_all.shape
        pred_n = normalize_activation(pred_all.reshape(B_ * T_, D_), mse_scale_f).reshape(B_, T_, D_)
        gold = torch.tensor(np.stack(golds), dtype=torch.float32, device=device)
        gold_n = normalize_activation(gold, mse_scale_f)  # [B,D]
        last = attn.sum(1) - 1  # [B]
        for d in range(max_dist + 1):
            pos = last - d
            ok = pos >= 0
            if int(ok.sum()) == 0:
                continue
            bi = torch.arange(B_, device=device)[ok]
            mse = ((pred_n[bi, pos[ok]] - gold_n[ok]) ** 2).mean(-1)  # [n]
            sse[d] += float(mse.sum()); cnt[d] += int(ok.sum())
    return [(d, (1.0 - (sse[d] / cnt[d]) / baseline) * 100.0 if cnt[d] > 0 else float("nan"), int(cnt[d]))
            for d in range(max_dist + 1)]


def ar_debug_stats(pred, gold, mse_scale_f):
    """Cheap per-batch AR diagnostics: norms + direction match (cosine).

    pred/gold are raw [B, d]. Returns dict for wandb. Helps tell apart "wrong
    scale" (norm mismatch) from "wrong direction" (low cosine) failures that a
    single normalized-MSE number hides.
    """
    with torch.no_grad():
        cos = F.cosine_similarity(pred.float(), gold.float(), dim=-1).mean().item()
        return {
            "pred_norm": pred.float().norm(dim=-1).mean().item(),
            "gold_norm": gold.float().norm(dim=-1).mean().item(),
            "cos_pred_gold": cos,
        }


@torch.no_grad()
def av_generate_samples(model, tokenizer, rows, cfg, device, *,
                        max_new_tokens=256, n_slots=1):
    """Generate explanations for a few fixed activations (AV debug table).

    Returns list of dicts: {idx, gen_len, explanation}.
    """
    model.eval()
    out = []
    vref = getattr(model, "_nla_vectors_ref", None)
    for i, row in enumerate(rows):
        msgs = [
            {**m, "content": m["content"].replace(INJECT_PLACEHOLDER, cfg.injection_char)}
            if isinstance(m.get("content"), str) else m
            for m in row["prompt"]
        ]
        ptxt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(ptxt, add_special_tokens=False)
        pt = torch.tensor([ids], dtype=torch.long, device=device)
        act = torch.tensor(row["activation_vector"], dtype=torch.float32)
        act = act.view(n_slots, -1).to(device)  # n_slots=1 -> [1, d], unchanged
        if vref is not None:
            vref[0] = act
        try:
            gen = model.generate(
                input_ids=pt, attention_mask=torch.ones_like(pt),
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, return_dict_in_generate=True,
            )
        finally:
            if vref is not None:
                vref[0] = None
        resp = tokenizer.decode(gen.sequences[0, pt.shape[1]:], skip_special_tokens=True)
        expl = extract_explanation(resp)
        out.append({
            "idx": i,
            "gen_len": int(gen.sequences.shape[1] - pt.shape[1]),
            "explanation": (expl or "<no extraction>")[:600],
        })
    model.train()
    return out


# ----------------------------------------------------------------------------
# AR critic init: truncate base Qwen3 to K+1 layers + Linear(d, d) value_head,
# identity-init the head. (Previously a separate critic-init script; now in-process.)
# ----------------------------------------------------------------------------

def _resolve_device_map(device_map_mode, max_gpu_mem, quant_config):
    """Return (device_map, max_memory) for from_pretrained.

    'single' → whole 4-bit model on GPU0 (bf16: None, caller does .to(device)).
    'auto'   → accelerate splits weights across visible GPUs (naive MP). A
               positive max_gpu_mem (GiB/GPU) forces a split — used to validate
               the 397B sharding path on a small model that would otherwise fit
               on one GPU.
    """
    if quant_config is None:
        return None, None
    if device_map_mode == "auto":
        max_memory = None
        if max_gpu_mem and max_gpu_mem > 0:
            max_memory = {
                i: f"{max_gpu_mem}GiB" for i in range(torch.cuda.device_count())
            }
        return "auto", max_memory
    return {"": 0}, None


def init_critic_from_base(base_ckpt: str, num_layers: int, dtype, quant_config=None,
                          device_map=None, max_memory=None, strip_final_norm=True):
    """Truncate base to first `num_layers` transformer blocks, attach an
    identity-init Linear(d, d) value_head. NLACriticModel handles the wrapping.

    identity-init is critical: at step 0, pred = value_head(last_h) = last_h
    when value_head = I, so the initial reconstruction loss starts at the
    backbone's own representational ceiling instead of `kaiming_uniform`'s
    1/√3 scaling which would crush pred_norm. See TRAINING_NOTES.md.

    quant_config (BitsAndBytesConfig) loads the backbone in 4-bit (QLoRA); the
    value_head stays full-precision (tiny, fully trainable).
    """
    # First load the full base, truncate the layers list, then construct
    # NLACriticModel around it.
    from copy import deepcopy
    base = AutoModelForCausalLM.from_pretrained(
        base_ckpt, torch_dtype=dtype, attn_implementation="sdpa",
        quantization_config=quant_config,
        device_map=device_map, max_memory=max_memory,
    )
    # Route through arch_adapters: multimodal wrappers (Gemma-3) expose the text
    # model under .language_model with config under .text_config, and GPT-2 /
    # Falcon keep decoder blocks at .transformer.h — the old bare
    # `while hasattr(.model)` walk crashed or truncated the wrong module there.
    from nla.utils.arch_adapters import resolve_text_model
    base = resolve_text_model(base)   # CausalLM-shaped text model (pass-through for Qwen/Llama)
    cfg = deepcopy(base.config)
    cfg.num_hidden_layers = num_layers
    if hasattr(cfg, "layer_types") and cfg.layer_types is not None:
        cfg.layer_types = list(cfg.layer_types)[:num_layers]
    # Inner decoder container: .model (llama family) or .transformer (GPT-2/Falcon),
    # holding the block list at .layers / .h respectively.
    if hasattr(base, "model"):
        inner, _layers_key = base.model, "layers"
    elif hasattr(base, "transformer"):
        inner, _layers_key = base.transformer, "h"
    else:
        raise AssertionError(
            f"{type(base).__name__} has neither .model nor .transformer — "
            f"extend init_critic_from_base for this architecture"
        )
    # Keep only the first num_layers blocks
    setattr(inner, _layers_key, torch.nn.ModuleList(
        list(cast(torch.nn.ModuleList, getattr(inner, _layers_key)))[:num_layers]))
    # Truncate the BACKBONE's own config too — NLACriticModel.save_pretrained
    # delegates to backbone.save_pretrained, which writes backbone.config.
    # Leaving it at the full depth makes a full (non-LoRA) AR save claim 36
    # layers with weights for 25; a later from_pretrained would then randomly
    # initialize the missing blocks and silently predict garbage.
    base.config.num_hidden_layers = num_layers
    if getattr(base.config, "layer_types", None) is not None:
        base.config.layer_types = list(base.config.layer_types)[:num_layers]
    if strip_final_norm:
        # RAW residual stream → value head. The full model's final
        # RMSNorm was trained for the LAST layer's output; applying it to the
        # layer-K stream bakes a per-channel γ reweighting into every critic
        # prediction. NLACriticModel.from_pretrained already strips it — this
        # makes the fresh-truncation path consistent. ar_meta.json records the
        # choice so RL reloads match (pre-2026-06 ckpts trained with norm kept).
        for _attr in ("norm", "final_layernorm", "ln_f", "final_layer_norm"):
            if hasattr(inner, _attr):
                setattr(inner, _attr, torch.nn.Identity())
                break
        else:
            raise AssertionError(
                f"could not find final layernorm on {type(inner).__name__}"
            )
    # lm_head is never used by the critic — drop it (frees ~1.2GB on 8B-class).
    if hasattr(base, "lm_head"):
        base.lm_head = torch.nn.Identity()
    d_model = cfg.hidden_size
    # NLACriticModel wraps backbone + value_head. Constructor takes both.
    critic = NLACriticModel(cfg, base)
    # Identity init the value head (Linear has bias=False per models.py:82)
    with torch.no_grad():
        critic.value_head.weight.copy_(torch.eye(d_model, dtype=dtype))
    # value_head stays FP32 regardless of backbone dtype: AdamW steps it directly
    # (no fp32 master), and in bf16 the identity diagonal (1.0, ULP≈0.0039)
    # can't absorb lr~1e-4 updates — they round to zero and the head never
    # moves. critic_predict casts in/out, so fp32 is compute-transparent.
    if quant_config is None:
        critic = critic.to(dtype)
        critic.value_head.to(dtype=torch.float32)
    else:
        # 4-bit backbone already placed (device_map); align value_head to the
        # LAST layer's device so forward's value_head(last_hidden) matches.
        last_dev = next(getattr(inner, _layers_key)[-1].parameters()).device
        critic.value_head.to(device=last_dev, dtype=torch.float32)
    print(f"[ar] truncated to {num_layers} layers, value_head identity-init "
          f"(weight norm = {critic.value_head.weight.float().norm().item():.3f})")
    return critic


def fresh_block_output_projs(block):
    """(name, nn.Linear) for the residual-branch OUTPUT projections of one
    decoder block: the token-mixer output proj (`o_proj` for attention,
    `out_proj` for gated-DeltaNet) + the MLP `down_proj`. Zeroing exactly these
    makes a standard pre-norm block (h += attn(norm(h)); h += mlp(norm(h))) an
    exact identity. Matches through peft's `.base_layer` wrapping so it works
    both before and after LoRA injection."""
    hits = []
    for n, m in block.named_modules():
        if not isinstance(m, torch.nn.Linear):
            continue
        base = n[: -len(".base_layer")] if n.endswith(".base_layer") else n
        if base.rsplit(".", 1)[-1] in ("o_proj", "out_proj", "down_proj"):
            hits.append((n, m))
    return hits


# ----------------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay to min_lr
# ----------------------------------------------------------------------------

def build_lr_lambda(warmup_steps, total_steps, min_lr_ratio):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, prog)
        cos = 0.5 * (1 + math.cos(math.pi * prog))
        return min_lr_ratio + (1 - min_lr_ratio) * cos
    return fn


# ----------------------------------------------------------------------------
# AV forward: encode chat-template prompt + response, build response-only loss
# mask, forward through model with Karvonen hook firing on the marker token.
# ----------------------------------------------------------------------------

@torch.no_grad()
def heldout_av_ce(model, tokenizer, rows, cfg, vectors_ref, device, *,
                  max_len=1024, micro_batch=16, n_slots=1):
    """Held-out AV val loss: mean token-CE on response tokens over doc-disjoint
    held-out AV rows — the SAME per-response-token CE the AV trains on, so it's
    directly comparable to the train `loss` (train loss is a memorization proxy;
    this is the generalization number). Returns (mean_ce, n_rows)."""
    tot_loss, tot_tok = 0.0, 0
    for cs in range(0, len(rows), micro_batch):
        chunk = rows[cs:cs + micro_batch]
        ids, attn, loss_mask, v_batch = _av_prepare_chunk(
            chunk, tokenizer, cfg.injection_char, device,
            max_len=max_len, n_slots=n_slots)
        vectors_ref[0] = v_batch
        try:
            logits = model(input_ids=ids, attention_mask=attn).logits.float()
        finally:
            vectors_ref[0] = None
        shift_logits = logits[:, :-1].contiguous()
        shift_targets = ids[:, 1:].to(shift_logits.device).contiguous()
        shift_mask = loss_mask[:, 1:].to(shift_logits.device).contiguous()
        per_tok = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1), reduction="none").view(shift_targets.shape)
        tot_loss += float((per_tok * shift_mask).sum().item())
        tot_tok += int(shift_mask.sum().item())
    return (tot_loss / max(tot_tok, 1)), len(rows)


def _av_prepare_chunk(rows, tokenizer, inject_char, device, max_len=1024,
                      n_slots=1):
    """Return (input_ids, attn, loss_mask, v_batch) — all [B, T] (or [B*n_slots, d]).

    n_slots > 1: each row's activation_vector holds n_slots*d floats (slot-major),
    the prompt contains n_slots consecutive markers, and v_batch is reshaped to
    [B*n_slots, d] — the row-major order karvonen_inject_in_residual consumes."""
    full_ids_list = []
    prompt_lens = []
    for row in rows:
        # row["prompt"] is list[{"role","content"}] with INJECT_PLACEHOLDER inside.
        # Replace with the actual injection char so the tokenizer emits the
        # marker token id at the right position.
        msgs = [
            {**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inject_char)}
            if isinstance(m.get("content"), str) else m
            for m in row["prompt"]
        ]
        prompt_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        # Response gets a trailing EOS so the model learns to stop.
        resp = row["response"] + (tokenizer.eos_token or "")
        resp_ids = tokenizer.encode(resp, add_special_tokens=False)
        full = prompt_ids + resp_ids
        if len(full) > max_len:
            # Truncate response from the right to fit. Prompt is fixed.
            full = full[:max_len]
        full_ids_list.append(torch.tensor(full, dtype=torch.long))
        prompt_lens.append(len(prompt_ids))

    bs = len(full_ids_list)
    T = max(t.numel() for t in full_ids_list)
    pad_id = tokenizer.eos_token_id
    batch_ids = torch.full((bs, T), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, T), dtype=torch.long, device=device)
    loss_mask = torch.zeros((bs, T), dtype=torch.float32, device=device)
    for i, t in enumerate(full_ids_list):
        L = t.numel()
        batch_ids[i, :L] = t.to(device)
        attn[i, :L] = 1
        # 1 on response positions, 0 on prompt + pad. The shift-by-one for CE
        # is applied later (in the loss computation), so this mask is in
        # "target token" space — positions whose CE we want to count.
        loss_mask[i, prompt_lens[i]:L] = 1
    v_batch = torch.tensor(
        np.stack([r["activation_vector"] for r in rows]),
        dtype=torch.float32, device=device,
    )
    if n_slots > 1:
        assert v_batch.shape[1] % n_slots == 0, (
            f"activation_vector dim {v_batch.shape[1]} not divisible by "
            f"n_slots={n_slots}"
        )
        v_batch = v_batch.view(bs * n_slots, v_batch.shape[1] // n_slots)
    return batch_ids, attn, loss_mask, v_batch


# ----------------------------------------------------------------------------
# AR forward: tokenize the already-built critic prompt, forward, take MSE on
# normalised (pred, gold).
# ----------------------------------------------------------------------------

def _ar_prepare_chunk(rows, tokenizer, device, max_len=1024):
    full_ids_list = []
    kept_rows = []
    n_skipped = 0
    for row in rows:
        # AR's prompt is the already-filled critic template string.
        # add_special_tokens=False matches RL-time critic scoring and stage-3's
        # build-time suffix verification (True is a no-op on Qwen but prepends
        # BOS on Llama/Gemma-family tokenizers → train/reward token mismatch).
        ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
        if len(ids) > max_len:
            # Right-truncating would cut the "</text> <summary>" suffix and the
            # last-token extraction would land mid-explanation — silently wrong.
            # Skip the row instead (RL-side rejects over-length the same way).
            n_skipped += 1
            continue
        full_ids_list.append(torch.tensor(ids, dtype=torch.long))
        kept_rows.append(row)
    if n_skipped:
        print(f"[ar] skipped {n_skipped}/{len(rows)} rows with critic prompt "
              f"> {max_len} tokens (suffix anchor would be truncated)")
    assert full_ids_list, f"all {len(rows)} rows exceeded max_len={max_len}"
    bs = len(full_ids_list)
    T = max(t.numel() for t in full_ids_list)
    pad_id = tokenizer.eos_token_id
    batch_ids = torch.full((bs, T), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, T), dtype=torch.long, device=device)
    for i, t in enumerate(full_ids_list):
        L = t.numel()
        batch_ids[i, :L] = t.to(device)
        attn[i, :L] = 1
    gold = torch.tensor(
        np.stack([r["activation_vector"] for r in kept_rows]),
        dtype=torch.float32, device=device,
    )
    return batch_ids, attn, gold


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    add_config_arg(p)
    p.add_argument("--mode", required=True, choices=["av", "ar"])
    p.add_argument("--base-ckpt", required=True,
                   help="HF dir for AV (base model) or AR (base model to truncate, "
                        "OR an already-prepared NLACriticModel checkpoint).")
    p.add_argument("--parquet", required=True, help="SFT data parquet")
    p.add_argument("--sidecar", default=None,
                   help="Sidecar source (defaults to --parquet for the dataset sidecar)")
    p.add_argument("--save-dir", required=True)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64,
                   help="Per-forward batch (= 'micro batch'). Effective batch = "
                        "batch_size × gradient_accumulation_steps.")
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--paged-optim", action="store_true", default=False,
                   help="Use bitsandbytes PagedAdamW8bit (optim states in CPU-pageable "
                        "memory) — frees ~35GB GPU for the full-FT AR critic to fit a 140GB H200.")
    p.add_argument("--ar-num-layers", type=int, default=None,
                   help="K+1 for AR mode — truncate base to this many transformer "
                        "blocks. Default: sidecar extraction.layer_index + 1 (falls "
                        "back to 25 = Qwen3-8B layer-24 if the sidecar lacks it); "
                        "an explicit value is asserted against the sidecar.")
    p.add_argument("--freeze-backbone", action="store_true", default=False,
                   help="AR mode: freeze the backbone and train ONLY the "
                        "value_head — a linear-probe baseline for how much of "
                        "the reconstruction is already linearly decodable.")
    p.add_argument("--ar-all-idx", action="store_true", default=False,
                   help="AR mode: DENSE objective — supervise the value head to reproduce the "
                        "target activation at EVERY position (each causal prefix), not just the "
                        "last/anchor token. 'reconstruct-as-you-read'; no privileged read point.")
    p.add_argument("--ar-fve-by-dist", action="store_true", default=False,
                   help="AR: at the final heldout, also log FVE reading at each distance-from-last "
                        "token (d0..d18) — the truncation-resistance curve.")
    p.add_argument("--ar-mlp-head", action="store_true", default=False,
                   help="AR mode: replace the Linear(d,d) value head with a deep pre-norm residual "
                        "MLP + identity affine (identity-init, co-trained) — tests whether the head "
                        "was the reconstruction bottleneck.")
    p.add_argument("--ar-mlp-hidden", type=int, default=16384, help="hidden dim of the residual-MLP head")
    p.add_argument("--ar-mlp-layers", type=int, default=4, help="number of residual-MLP blocks")
    p.add_argument("--ar-summary-token", action="store_true", default=False,
                   help="AR mode: register a dedicated <|summary|> special token as the read "
                        "anchor (data prompts must END with it). Trains ONLY {LoRA adapters + "
                        "value_head + the new token's embedding row} — a grad hook zeroes the "
                        "embedding gradient for every other row. Tests whether a purpose-built "
                        "single aggregation token beats the multi-bpe '</text> <summary>' anchor.")
    p.add_argument("--ar-fresh-block", action="store_true", default=False,
                   help="AR mode: keep ONE structural decoder block ABOVE the extraction "
                        "layer (--ar-num-layers = sidecar layer_index + 2), re-initialize "
                        "it to an EXACT identity (zero its attention/deltanet output "
                        "projection + MLP down projection so both residual branches emit "
                        "0), and train the whole block (in addition to LoRA + value_head). "
                        "Gives the read position a trainable cross-position aggregation "
                        "step that per-position heads (--ar-mlp-head) lack. At init the "
                        "block is a passthrough, so step-0 metrics must match the "
                        "plain-identity-head baseline's.")
    # ---- Debug sampling: periodically dump example generations to a wandb Table ----
    p.add_argument("--sample-every", type=int, default=0,
                   help="Every N steps, log example generations to an accumulating "
                        "wandb Table (AV: generate explanations from injected "
                        "activations; AR: reconstruct). 0 = off.")
    p.add_argument("--n-samples", type=int, default=4,
                   help="Number of fixed examples per --sample-every dump.")
    p.add_argument("--sample-max-new-tokens", type=int, default=256,
                   help="AV sampling: max new tokens when generating explanations.")
    p.add_argument("--heldout-parquet", default=None,
                   help="AR mode: AV-split parquet for held-out FVE (doc-disjoint "
                        "from AR training data by stage-1 construction). "
                        "AV mode: held-out AV parquet (prompt/response/activation) "
                        "for inline val token-CE + ppl (the generalization metric, "
                        "vs the memorization-proxy train loss). "
                        "Evaluated every --heldout-every steps.")
    p.add_argument("--heldout-rows", type=int, default=1000)
    p.add_argument("--heldout-every", type=int, default=100)
    p.add_argument("--strip-final-norm", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="AR mode: replace the backbone's final RMSNorm with "
                        "Identity so the value head sees the raw layer-K "
                        "residual (matches NLACriticModel."
                        "from_pretrained). --no-strip-final-norm reproduces "
                        "pre-2026-06 checkpoints. Recorded in ar_meta.json.")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--n-slots", type=int, default=1,
                   help="AV multi-slot injection: activation_vector holds "
                        "n_slots*d floats (slot-major) and the prompt contains "
                        "n_slots consecutive markers; each slot is Karvonen-"
                        "injected at its own marker position")
    p.add_argument("--lr", type=float, default=None,
                   help="If omitted: AV-mode default 1e-4 (best for a 1-epoch warm-start "
                        "in our sweeps), AR-mode default 2e-5.")
    p.add_argument("--min-lr", type=float, default=2e-6)
    p.add_argument("--lr-warmup-steps", type=int, default=50)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Default: ON for AV (fits 8B + batch=64 + FA2 on 141 GB H200), "
                        "OFF for AR (smaller model + shorter seq fits comfortably).")
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--full-ft-dtype", choices=["fp32", "bf16"], default="fp32",
                   help="Parameter dtype for FULL fine-tuning (no LoRA). fp32 keeps the "
                        "weights in fp32 and autocasts compute to bf16: in pure bf16 an "
                        "Adam update of ~lr rounds to ZERO on any weight with |w| > lr*512 "
                        "(round-to-nearest ULP), so at the AR default lr=2e-5 ~60%% of a "
                        "N(0,0.02) backbone and every ~1.0-scale norm weight silently never "
                        "move. Costs 2x weight+grad memory (checkpoints also save fp32; "
                        "downstream loaders pass torch_dtype and cast back). Pass bf16 to "
                        "trade correctness for memory. LoRA/4-bit paths ignore this — "
                        "adapters start near zero, where bf16 ULP is fine.")
    p.add_argument("--quant", choices=["none", "4bit"], default="none",
                   help="4bit = bitsandbytes nf4 (QLoRA). Required for models too "
                        "big for bf16; validates the GLM-5 path on Qwen3-8B.")
    p.add_argument("--use-lora", action="store_true", default=False,
                   help="Train a LoRA adapter on a frozen base instead of full-FT. "
                        "Mandatory for 4bit. (AR value_head stays fully trainable.)")
    p.add_argument("--lora-r", type=int, default=128)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-scope", choices=["attn", "all"], default="attn",
                   help="AV LoRA target modules. 'attn' (default) = token-mixing only; "
                        "'all' additionally adapts the MLP (+ deltanet decay projs for "
                        "qwen3_5) — a more expressive verbalizer. AR keeps attn-only "
                        "(the deliberate reconstruction bottleneck).")
    p.add_argument("--device-map", choices=["single", "auto"], default="single",
                   help="single = whole 4-bit model on GPU0 (fits up to ~70B on "
                        "a B200). auto = accelerate splits weights across all "
                        "visible GPUs (naive MP) — required for 397B-class bases.")
    p.add_argument("--max-gpu-mem", type=int, default=0,
                   help="GiB/GPU cap for device_map=auto weight placement. >0 "
                        "forces a multi-GPU split (used to validate sharding on a "
                        "small model). 0 = use full GPU memory.")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap training rows (smoke runs)")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="nla-qwen3-8b")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-group", default="warmstart",
                   help="wandb group for organizing the workspace (warmstart/rl/eval).")
    p.add_argument("--wandb-tags", default=None,
                   help="comma-separated wandb tags for explicit experiments (e.g. 'sweep,lr3e5').")
    p.add_argument("--no-wandb", action="store_true")
    apply_config_defaults(p)   # YAML (--config) -> argparse defaults; CLI still overrides
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    dtype = torch.bfloat16
    if args.lr is None:
        # Mode-aware default: non-comp AV warmstart is 1e-4 (2x-data 1-epoch best, held-out
        # val ppl 3.86; optimum dropped from the old 1x 2e-4 after the data doubled);
        # AR keeps 2e-5 (AR launchers pass --lr explicitly anyway).
        args.lr = 1e-4 if args.mode == "av" else 2e-5
    if args.gradient_checkpointing is None:
        args.gradient_checkpointing = (args.mode == "av")
    # fp32 master weights for full fine-tuning (see --full-ft-dtype help).
    full_ft = (not args.use_lora) and args.quant != "4bit"
    param_dtype = torch.float32 if (full_ft and args.full_ft_dtype == "fp32") else dtype
    amp_enabled = param_dtype is torch.float32

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled)

    if amp_enabled:
        print("[dtype] full-FT: fp32 params + bf16 autocast compute "
              "(--full-ft-dtype bf16 to disable)")
    if args.sidecar is None:
        args.sidecar = args.parquet

    # ---- tokenizer + nla config ----
    # From --base-ckpt, NOT hardcoded — the sidecar asserts below catch a
    # wrong-family tokenizer, but only if we load the one the run targets.
    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)
    # --ar-summary-token: register the dedicated read-anchor token BEFORE any
    # data tokenization so encode() emits its id wherever the literal string
    # "<|summary|>" appears in the critic prompts (add_special_tokens=False
    # still resolves REGISTERED special tokens found in the text).
    sumtok_id = None
    if args.ar_summary_token:
        assert args.mode == "ar", "--ar-summary-token is AR-only"
        _n_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|summary|>"]})
        sumtok_id = tokenizer.convert_tokens_to_ids("<|summary|>")
        assert isinstance(sumtok_id, int) and sumtok_id >= 0
        print(f"[sumtok] registered <|summary|> (added={_n_added}) -> id {sumtok_id} "
              f"(len(tokenizer)={len(tokenizer)})", flush=True)
    cfg = load_nla_config(args.sidecar, tokenizer)
    mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
    print(f"[cfg] mode={args.mode} d_model={cfg.d_model} mse_scale={mse_scale_f}")
    # AR depth comes from the DATA, not a magic number: activations extracted at
    # layer K must be reconstructed by a K+1-block critic. 25 was a Qwen3-8B
    # (layer-24) constant that silently mistrained on any other extraction layer.
    assert not (args.ar_fresh_block and args.mode != "ar"), "--ar-fresh-block is AR-only"
    if args.mode == "ar":
        _side_k = cfg.extraction_layer_index
        # --ar-fresh-block keeps ONE extra block ABOVE the extraction layer
        # (identity-re-init at build time below), so the required depth is
        # layer_index + 2: blocks 0..K reproduce the extraction stream, block
        # K+1 is the fresh trainable aggregator.
        _fb_extra = 1 if args.ar_fresh_block else 0
        if args.ar_num_layers is None:
            args.ar_num_layers = (_side_k + 1 + _fb_extra) if _side_k is not None else 25 + _fb_extra
            print(f"[ar] --ar-num-layers defaulted to {args.ar_num_layers} "
                  f"({'sidecar layer_index+1' if _side_k is not None else 'no sidecar layer_index; Qwen3-8B fallback'}"
                  f"{' + 1 fresh block' if _fb_extra else ''})")
        elif _side_k is not None:
            assert args.ar_num_layers == _side_k + 1 + _fb_extra, (
                f"--ar-num-layers {args.ar_num_layers} != sidecar "
                f"extraction.layer_index+1{'+1 (--ar-fresh-block)' if _fb_extra else ''} "
                f"= {_side_k + 1 + _fb_extra} — the critic would "
                f"read a different layer than the activations were captured at."
            )

    # ---- model ----
    sumtok_emb_weight = None   # set on the AR path when --ar-summary-token
    if args.mode == "av":
        print(f"[av] loading {args.base_ckpt} (quant={args.quant}, lora={args.use_lora})")
        quant_config = None
        if args.quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=dtype,  # FSDP-friendly storage (harmless single-GPU)
            )
        dmap, max_mem = _resolve_device_map(args.device_map, args.max_gpu_mem, quant_config)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_ckpt, torch_dtype=param_dtype,
            attn_implementation=args.attn_implementation,
            quantization_config=quant_config,
            device_map=dmap, max_memory=max_mem,
        )
        if dmap is None:
            model = model.to(device)
        elif args.device_map == "auto" and hasattr(model, "hf_device_map"):
            print(f"[av] device_map=auto → GPUs used: "
                  f"{sorted({d for d in model.hf_device_map.values() if isinstance(d, int)})}")
        if args.use_lora:
            if quant_config is not None:
                model = prepare_model_for_kbit_training(
                    model, use_gradient_checkpointing=args.gradient_checkpointing,
                )
            from nla.utils.arch_adapters import resolve_lora_target_modules
            _tm = resolve_lora_target_modules(model.config, args.lora_scope)
            print(f"[av] LoRA scope={args.lora_scope} → {len(_tm)} module types: {_tm}")
            model = get_peft_model(model, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                bias="none", task_type="CAUSAL_LM", use_rslora=True,
                target_modules=_tm,
            ))
            model.print_trainable_parameters()
        vectors_ref = [None]
        register_karvonen_hook(
            model, vectors_ref,
            cfg.injection_token_id,
            cfg.injection_left_neighbor_id,
            cfg.injection_right_neighbor_id,
        )
        model._nla_vectors_ref = vectors_ref  # av_generate_samples reaches it here
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            print("[av] gradient_checkpointing ENABLED")
    else:  # ar
        quant_config = None
        if args.quant == "4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=dtype,
            )
        dmap, max_mem = _resolve_device_map(args.device_map, args.max_gpu_mem, quant_config)
        # Check if --base-ckpt is already a critic ckpt (has value_head.safetensors)
        is_prepared_critic = (Path(args.base_ckpt) / "value_head.safetensors").exists()
        if is_prepared_critic:
            print(f"[ar] loading pre-prepared critic from {args.base_ckpt}")
            model = NLACriticModel.from_pretrained(
                args.base_ckpt, torch_dtype=param_dtype,
                attn_implementation=args.attn_implementation,
                quantization_config=quant_config,
                device_map=dmap, max_memory=max_mem,
            )
            if dmap is None:
                model = model.to(device)
            # from_pretrained ALWAYS strips the final norm, regardless of the
            # CLI flag — record what actually happened, or RL would rebuild
            # the critic differently than it was trained.
            if not args.strip_final_norm:
                print("[ar] NOTE: --no-strip-final-norm ignored on the "
                      "prepared-critic path (from_pretrained always strips); "
                      "recording final_norm_stripped=true")
            args.strip_final_norm = True
        else:
            print(f"[ar] truncating base {args.base_ckpt} to {args.ar_num_layers} "
                  f"layers (quant={args.quant})")
            model = init_critic_from_base(
                args.base_ckpt, args.ar_num_layers, param_dtype, quant_config,
                device_map=dmap, max_memory=max_mem,
                strip_final_norm=args.strip_final_norm,
            )
            if dmap is None:
                model = model.to(device)
        fresh_block = None
        if args.ar_fresh_block:
            assert not is_prepared_critic, (
                "--ar-fresh-block needs a fresh truncation from the base model, "
                "not a pre-prepared critic checkpoint (its extra block would "
                "already be trained, not identity)")
            from nla.utils.arch_adapters import resolve_decoder_layers
            _dl = resolve_decoder_layers(model.backbone)
            assert len(_dl) == args.ar_num_layers, (len(_dl), args.ar_num_layers)
            fresh_block = _dl[-1]
            # Zero BOTH residual branches' output projections → the block is an
            # EXACT identity passthrough at init; the rest of its (pretrained)
            # weights stay as-is and train back in via gradients.
            _zeroed = []
            with torch.no_grad():
                for _n, _m in fresh_block_output_projs(fresh_block):
                    _m.weight.zero_()
                    if _m.bias is not None:
                        _m.bias.zero_()
                    _zeroed.append(_n)
            assert any(_n.rsplit(".", 1)[-1] in ("o_proj", "out_proj") for _n in _zeroed) and \
                any(_n.rsplit(".", 1)[-1] == "down_proj" for _n in _zeroed), (
                f"fresh block: expected a token-mixer output proj AND an MLP "
                f"down_proj to zero; got {_zeroed} — extend "
                f"fresh_block_output_projs for this architecture")
            print(f"[ar] fresh block = layer {args.ar_num_layers - 1} "
                  f"(type={getattr(fresh_block, 'block_type', '?')}); zeroed "
                  f"output projs {_zeroed} → exact identity at init", flush=True)
            # Hard self-test: the block's recorded output must EQUAL its input
            # bit-for-bit on a real forward (0-weight matmul is exactly 0, and
            # residual + 0 is exact in any float dtype).
            from nla.models import _inner_transformer
            model.eval()
            with torch.no_grad():
                _dev = next(fresh_block.parameters()).device
                _tids = torch.randint(100, 1000, (2, 16), device=_dev)
                _hs = _inner_transformer(model.backbone)(
                    input_ids=_tids, attention_mask=torch.ones_like(_tids),
                    output_hidden_states=True, use_cache=False).hidden_states
                _delta = (_hs[-1] - _hs[-2]).abs().max().item()
                assert torch.equal(_hs[-1], _hs[-2]), (
                    f"fresh block is NOT an exact identity at init: its output "
                    f"differs from its input (max |Δ| = {_delta:.3e}) — wrong "
                    f"projections zeroed?")
            print("[ar] fresh-block identity self-test PASSED "
                  "(block output == block input exactly on a random batch)", flush=True)
        if args.use_lora:
            # Inject LoRA IN-PLACE into the backbone's attn projections. Unlike
            # get_peft_model this does NOT wrap the backbone in a PeftModel, so
            # NLACriticModel.forward (which calls the inner transformer directly)
            # is unchanged and the value_head stays a plain trainable module.
            from peft import inject_adapter_in_model
            if quant_config is not None:
                model.backbone = prepare_model_for_kbit_training(
                    model.backbone, use_gradient_checkpointing=args.gradient_checkpointing,
                )
            from nla.utils.arch_adapters import resolve_attn_target_modules, resolve_lora_target_modules
            _ar_tm = resolve_lora_target_modules(model.backbone.config, args.lora_scope)
            print(f"[ar] LoRA scope={args.lora_scope} -> {len(_ar_tm)} module types: {_ar_tm}", flush=True)
            inject_adapter_in_model(LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
                bias="none", task_type="CAUSAL_LM", use_rslora=True,
                target_modules=_ar_tm,
            ), model.backbone)
            # bf16 LoRA + AdamW rounds sub-ULP updates to zero (measured ~86% adapter freeze
            # late in long runs); keep adapters in fp32 so the updates actually land.
            _n_fp32 = 0
            for _n, _p in model.backbone.named_parameters():
                if "lora_" in _n and _p.dtype != torch.float32:
                    _p.data = _p.data.float(); _n_fp32 += 1
            print(f"[ar] cast {_n_fp32} LoRA adapter tensors to fp32 (bf16-ULP-freeze fix)", flush=True)
            if args.ar_mlp_head:
                _hd = next(model.value_head.parameters()).device
                model.value_head = ResidualMLPHead(
                    cfg.d_model, args.ar_mlp_hidden, args.ar_mlp_layers).to(device=_hd, dtype=torch.float32)
                print(f"[ar] value_head -> ResidualMLPHead(d={cfg.d_model} hidden={args.ar_mlp_hidden} "
                      f"layers={args.ar_mlp_layers}) identity-init fp32 "
                      f"({sum(p.numel() for p in model.value_head.parameters())/1e6:.0f}M params)", flush=True)
            # Train ONLY the LoRA adapters + the value_head; freeze the rest.
            for n_, p_ in model.named_parameters():
                p_.requires_grad_(("lora_" in n_) or n_.startswith("value_head"))
            n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[ar] LoRA-injected; trainable={n_tr/1e6:.1f}M (lora + value_head)")
        vectors_ref = None
        if args.gradient_checkpointing and not args.use_lora:
            # NLACriticModel wraps backbone; enable on inner module
            # (use_lora+4bit path already enabled it via prepare_model_for_kbit_training)
            if hasattr(model.backbone, "gradient_checkpointing_enable"):
                model.backbone.gradient_checkpointing_enable()
                print("[ar] gradient_checkpointing ENABLED (backbone)")
        if args.freeze_backbone:
            # Linear-probe baseline: freeze the backbone, train ONLY the
            # value_head. (The old semantics froze everything incl. the head,
            # which then tripped the no-trainable-params assert below — the
            # flag was unusable.)
            for p_ in model.parameters():
                p_.requires_grad_(False)
            for p_ in model.value_head.parameters():
                p_.requires_grad_(True)
            print("[ar] backbone FROZEN — training value_head only "
                  "(linear-probe baseline, --freeze-backbone)")
        if args.ar_fresh_block:
            # Runs AFTER the LoRA/freeze requires_grad loops (they froze the
            # block) and BEFORE the optimizer gathers trainable params: the
            # fresh block trains FULLY, on top of the LoRA adapters + value_head.
            for p_ in fresh_block.parameters():
                p_.requires_grad_(True)
            # bf16 params + direct (no-master-copy) Adam updates round ~every
            # update on pretrained-magnitude weights to zero (ULP(w)/2 > lr for
            # |w| > lr*512 — see --full-ft-dtype help), which would silently
            # freeze most of the block. Keep the block's params fp32 and run its
            # forward under a block-local bf16 autocast: layers 0..K keep the
            # baseline's exact bf16 numerics, the block computes in bf16, and
            # grads/updates accumulate in fp32.
            fresh_block.to(torch.float32)
            _fb_fwd = fresh_block.forward
            def _fb_autocast_forward(*_a, **_kw):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    return _fb_fwd(*_a, **_kw)
            fresh_block.forward = _fb_autocast_forward
            _n_fb = sum(p_.numel() for p_ in fresh_block.parameters())
            print(f"[ar] fresh block (layer {args.ar_num_layers - 1}) FULLY "
                  f"trainable: {_n_fb / 1e6:.0f}M params, fp32 + block-local "
                  f"bf16 autocast", flush=True)
        if args.ar_summary_token:
            # Give <|summary|> an embedding row and make ONLY that row learn.
            # Runs AFTER the LoRA/freeze requires_grad loops (they froze the
            # embedding) and BEFORE the optimizer gathers requires_grad params.
            emb = model.backbone.get_input_embeddings()
            V_old, d_emb = emb.weight.shape
            if sumtok_id >= V_old:
                # backbone.resize_token_embeddings would also try to resize the
                # output head, which is nn.Identity here (lm_head stripped) —
                # build the enlarged input embedding manually instead.
                new_emb = torch.nn.Embedding(
                    sumtok_id + 1, d_emb,
                    dtype=emb.weight.dtype, device=emb.weight.device)
                with torch.no_grad():
                    new_emb.weight[:V_old] = emb.weight
                    new_emb.weight[V_old:] = emb.weight.mean(dim=0, keepdim=True)
                model.backbone.set_input_embeddings(new_emb)
                model.backbone.config.vocab_size = sumtok_id + 1
                model.config.vocab_size = sumtok_id + 1
                emb = model.backbone.get_input_embeddings()
                print(f"[sumtok] input embeddings resized {V_old} -> {sumtok_id + 1} "
                      f"(new row init = mean of existing rows)", flush=True)
            else:
                # padded vocab: the model already has a (untrained) row at this id
                with torch.no_grad():
                    emb.weight[sumtok_id] = emb.weight[:V_old].mean(dim=0)
                print(f"[sumtok] id {sumtok_id} < embedding rows {V_old} (padded "
                      f"vocab) — no resize; row re-init = mean of existing rows",
                      flush=True)
            # Autograd needs the whole leaf trainable; a grad hook zeroes every
            # row except the new token's so nothing else moves (weight_decay=0,
            # and Adam moments stay exactly 0 where grads are exactly 0).
            emb.weight.requires_grad_(True)
            sumtok_emb_weight = emb.weight
            _sumtok_hook_dbg = [True]

            def _sumtok_grad_mask(grad, _tid=sumtok_id, _dbg=_sumtok_hook_dbg):
                out = torch.zeros_like(grad)
                out[_tid] = grad[_tid]
                if _dbg[0]:
                    _dbg[0] = False
                    nz_in = int((grad.abs().sum(dim=1) > 0).sum().item())
                    nz_out = (out.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
                    print(f"[sumtok] grad-mask hook (first backward): incoming rows "
                          f"w/ nonzero grad = {nz_in}; after mask = {nz_out} "
                          f"(expect [{_tid}]); |g[{_tid}]| = "
                          f"{out[_tid].norm().item():.3e}", flush=True)
                return out

            emb.weight.register_hook(_sumtok_grad_mask)
            print(f"[sumtok] embedding row {sumtok_id} TRAINABLE "
                  f"(grad hook masks all other rows)", flush=True)
    model.train()

    # ---- data ----
    print(f"[data] loading {args.parquet} (max_rows={args.max_rows})", flush=True)
    rows = load_sft_dataset(args.parquet, n_max=args.max_rows, mode=args.mode)
    print(f"[data] {len(rows)} rows", flush=True)
    ar_prefix_len = 0
    if args.mode == "ar" and args.ar_all_idx:
        # dense objective supervises every position from the explanation start on
        # (skip the fixed template prefix before {explanation}, which carries no signal).
        _prefix = (cfg.critic_prompt_template or "{explanation}").split("{explanation}")[0]
        ar_prefix_len = len(tokenizer.encode(_prefix, add_special_tokens=False))
        print(f"[ar] ALL-IDX dense supervision; skipping first {ar_prefix_len} prefix tokens", flush=True)
    if args.mode == "ar" and args.ar_summary_token:
        # Replaces the critic_suffix_ids check (sidecar sets it null for this
        # variant): the tokenized prompt must END with the <|summary|> id, or
        # last-token extraction reads the wrong position.
        _ids0 = tokenizer.encode(rows[0]["prompt"], add_special_tokens=False)
        assert _ids0 and _ids0[-1] == sumtok_id, (
            f"sumtok: rows[0] tokenized tail {_ids0[-4:]} does not end with "
            f"<|summary|> id {sumtok_id} — data prompts must end with the "
            f"literal '<|summary|>' string")
        print(f"[sumtok] rows[0] tail ids {_ids0[-4:]} -> "
              f"{tokenizer.convert_ids_to_tokens(_ids0[-4:])} "
              f"(last == <|summary|> ✓)", flush=True)
        if cfg.critic_prompt_template is not None:
            _tids = tokenizer.encode(
                cfg.critic_prompt_template.format(explanation="x"),
                add_special_tokens=False)
            assert _tids[-1] == sumtok_id, (
                f"sidecar ar template must end with <|summary|> "
                f"(heldout eval uses it); got last id {_tids[-1]}")
            print(f"[sumtok] heldout template ends with <|summary|> ✓", flush=True)
    if args.mode == "ar" and cfg.critic_suffix_ids:
        # One-time suffix-anchor sanity check (the sidecar field's stated
        # purpose): the tokenized critic prompt must end with the expected
        # "</text> <summary>" ids, or last-token extraction trains on the
        # wrong position. Row 0 suffices — template drift hits every row.
        from nla.config import verify_critic_suffix
        _row0_ids = tokenizer.encode(rows[0]["prompt"], add_special_tokens=False)
        verify_critic_suffix(_row0_ids, cfg.critic_suffix_ids, context="ar row 0")
        print(f"[ar] critic suffix anchor verified (row 0)")

    # ---- optimizer + LR schedule ----
    try:
        import bitsandbytes as bnb
        if getattr(args, "paged_optim", False):
            optim_cls = bnb.optim.PagedAdamW8bit  # optim states in CPU-pageable memory
            print(f"[optim] using bitsandbytes PagedAdamW8bit (CPU-paged states) (bnb {bnb.__version__})")
        else:
            optim_cls = bnb.optim.AdamW8bit
            print(f"[optim] using bitsandbytes AdamW8bit (bnb {bnb.__version__})")
    except ImportError:
        optim_cls = torch.optim.AdamW
        print("[optim] bitsandbytes unavailable, falling back to torch AdamW (fp32 m,v)")
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, (
        "no trainable parameters — --freeze-backbone freezes the whole AR, so "
        "there is nothing to optimize in AR-SFT. Drop --freeze-backbone."
    )
    if sumtok_emb_weight is not None:
        assert any(p_ is sumtok_emb_weight for p_ in trainable), (
            "sumtok: embedding weight missing from the optimizer's trainable set — "
            "requires_grad flip must run before this point")
        print(f"[sumtok] embedding weight IS in optimizer trainables "
              f"({tuple(sumtok_emb_weight.shape)}; only row {sumtok_id} receives grad)",
              flush=True)
    optim = optim_cls(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        build_lr_lambda(args.lr_warmup_steps, args.num_steps,
                        args.min_lr / max(args.lr, 1e-12)),
    )
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[optim] trainable params: {n_trainable / 1e9:.2f} B")

    # ---- AR-only: predict-the-mean baseline for FVE logging ----
    # Paper definition: baseline = E[||v_norm - μ||²] (raw variance of the
    # normalized distribution, ≈0.72), NOT MSE against normalize(μ) (≈0.94)
    # which runs before 2026-06-09 used and which inflates FVE.
    fve_baseline = None
    if args.mode == "ar":
        from nla.schema import compute_predict_mean_baselines
        _act = torch.tensor(
            np.stack([r["activation_vector"] for r in rows[: min(len(rows), 4000)]]),
            dtype=torch.float32,
        )
        _bl_meannorm, fve_baseline = compute_predict_mean_baselines(_act, mse_scale_f)
        print(f"[ar] predict-the-mean MSE baseline = {fve_baseline:.4f} "
              f"(paper def; meannorm baseline = {_bl_meannorm:.4f})")

    # ---- AR-only: held-out FVE pairs (doc-disjoint AV split) ----
    heldout_pairs = None
    heldout_baseline = None
    heldout_av_rows = None
    if args.mode == "av" and args.heldout_parquet:
        heldout_av_rows = load_sft_dataset(
            args.heldout_parquet, args.heldout_rows, mode="av")
        print(f"[av] {len(heldout_av_rows)} held-out AV rows from {args.heldout_parquet} "
              f"-> inline val token-CE/ppl every {args.heldout_every} steps", flush=True)
    if args.mode == "ar" and args.heldout_parquet:
        assert cfg.critic_prompt_template is not None, (
            "--heldout-parquet needs critic_prompt_template in the sidecar"
        )
        heldout_pairs = load_heldout_explanation_pairs(
            args.heldout_parquet, args.heldout_rows,
        )
        _h_acts = torch.tensor(
            np.stack([a for _, a in heldout_pairs]), dtype=torch.float32,
        )
        _, heldout_baseline = compute_predict_mean_baselines(_h_acts, mse_scale_f)
        del _h_acts
        print(f"[ar] {len(heldout_pairs)} held-out pairs from "
              f"{args.heldout_parquet}; baseline (paper def) = {heldout_baseline:.4f}")

    # ---- wandb ----
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_name,
                   group=args.wandb_group,
                   tags=(args.wandb_tags.split(",") if args.wandb_tags else None),
                   config=vars(args))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(args, save_dir)   # snapshot merged config for reproducibility

    # ---- debug sampling: fixed example set + accumulating table ----
    sample_rows = rows[: args.n_samples] if args.sample_every > 0 else []
    sample_table_data = []

    # ---- training loop ----
    _sumtok_row0 = (sumtok_emb_weight[sumtok_id].detach().float().clone()
                    if sumtok_emb_weight is not None else None)
    _sumtok_nbr0 = (sumtok_emb_weight[sumtok_id - 1].detach().float().clone()
                    if sumtok_emb_weight is not None else None)
    _sumtok_moved = [False]
    if _sumtok_row0 is not None:
        print(f"[sumtok] emb row init norm = {_sumtok_row0.norm().item():.3e} "
              f"(mean |elem| = {_sumtok_row0.abs().mean().item():.3e})", flush=True)
    rng = np.random.default_rng(args.seed)
    perm = list(range(len(rows)))
    rng.shuffle(perm)
    cursor = 0

    grad_accum = args.gradient_accumulation_steps
    eff_batch = args.batch_size * grad_accum
    print(f"[loop] {args.num_steps} steps, batch={args.batch_size} × "
          f"grad_accum={grad_accum} = eff_batch={eff_batch}")

    for step in range(args.num_steps):
        t0 = time.time()
        optim.zero_grad()
        accum_loss = 0.0
        accum_resp_tokens = 0  # AV only: total response tokens for normalization
        accum_av_entropy = 0.0  # AV only: mean policy entropy over response tokens (nats)
        accum_n = 0
        ar_dbg = {}            # AR: last-chunk norms/cosine snapshot

        for accum_idx in range(grad_accum):
            # ---- pick batch ----
            if cursor + args.batch_size > len(perm):
                rng.shuffle(perm)
                cursor = 0
            chunk_rows = [rows[i] for i in perm[cursor:cursor + args.batch_size]]
            cursor += args.batch_size

            # ---- forward + loss ----
            if args.mode == "av":
                ids, attn, loss_mask, v_batch = _av_prepare_chunk(
                    chunk_rows, tokenizer, cfg.injection_char, device,
                    max_len=args.max_len, n_slots=args.n_slots,
                )
                # vectors_ref stays set through .backward() below: AV mode runs
                # gradient checkpointing BY DEFAULT, the backward-time recompute
                # re-fires the injection hook, and clearing before backward made
                # the recompute skip the injection's Jacobian — a silent gradient
                # error on the marker pathway (verified vs no-checkpoint grads).
                vectors_ref[0] = v_batch
                with amp():
                    logits = model(input_ids=ids, attention_mask=attn).logits
                logits = logits.float()
                # Shift-by-one CE on response tokens. Predict ids[:, t+1] from
                # logits[:, t]. Mask is in TARGET space (positions of tokens
                # to predict), so mask[:, 1:] aligned with logits[:, :-1].
                shift_logits = logits[:, :-1].contiguous()
                # device_map=auto can return logits on a non-zero GPU; align.
                shift_targets = ids[:, 1:].to(shift_logits.device).contiguous()
                shift_mask = loss_mask[:, 1:].to(shift_logits.device).contiguous()
                V = shift_logits.size(-1)
                per_tok = F.cross_entropy(
                    shift_logits.view(-1, V),
                    shift_targets.view(-1),
                    reduction="none",
                ).view(shift_targets.shape)
                n_resp = shift_mask.sum().clamp(min=1)
                loss = (per_tok * shift_mask).sum() / n_resp
                accum_resp_tokens += int(n_resp.item())
                # mean token entropy over response positions (nats), logging only.
                # Gather response tokens first so the softmax is over n_resp rows, not B*T.
                with torch.no_grad():
                    resp_logits = shift_logits[shift_mask.bool()]   # [n_resp, V]
                    lsm = F.log_softmax(resp_logits, dim=-1)
                    accum_av_entropy += float((-(lsm.exp() * lsm).sum(-1)).mean())
            else:  # ar, single-vector
                ids, attn, gold = _ar_prepare_chunk(
                    chunk_rows, tokenizer, device, max_len=args.max_len,
                )
                gold_n = normalize_activation(gold, mse_scale_f)
                if args.ar_all_idx:
                    # dense: supervise reconstruction at EVERY position (causal prefix -> target).
                    with amp():
                        pred_all = critic_predict_all(model, ids, attn, mse_scale_f)  # [B,T,D]
                    B_, T_, D_ = pred_all.shape
                    pred_all_n = normalize_activation(
                        pred_all.reshape(B_ * T_, D_), mse_scale_f).reshape(B_, T_, D_)
                    mse_pos = ((pred_all_n - gold_n[:, None, :]) ** 2).mean(-1)  # [B,T]
                    sup = attn.float().clone()
                    sup[:, :ar_prefix_len] = 0.0
                    loss = (mse_pos * sup).sum() / sup.sum().clamp(min=1)
                    _li = attn.sum(1) - 1                    # last real token = eval read point
                    pred_last = pred_all[torch.arange(B_, device=pred_all.device), _li]
                    ar_dbg = ar_debug_stats(pred_last, gold, mse_scale_f)
                else:
                    with amp():
                        pred = critic_predict(model, ids, attn, mse_scale_f)
                    pred_n = normalize_activation(pred, mse_scale_f)
                    loss = F.mse_loss(pred_n, gold_n)
                    ar_dbg = ar_debug_stats(pred, gold, mse_scale_f)

            # Scale loss for accumulation; gradients sum correctly.
            try:
                (loss / grad_accum).backward()
            finally:
                if vectors_ref is not None:   # AR mode has no injection hook
                    vectors_ref[0] = None   # clear only AFTER backward (checkpoint recompute done)
            accum_loss += loss.item()
            accum_n += 1

        # ---- step ----
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if args.ar_fresh_block and step == 0:
            _nz = [n_ for n_, p_ in fresh_block.named_parameters()
                   if p_.grad is not None and float(p_.grad.abs().max()) > 0]
            print(f"[ar] fresh block @step0: {len(_nz)} param tensors with "
                  f"nonzero grad: {_nz} (zero-init output projs gate gradient "
                  f"flow to the rest of the block until they move off 0)", flush=True)
        optim.step()
        sched.step()
        if args.ar_fresh_block and ((step + 1) in (3, 10, 20, 50) or (step + 1) % 150 == 0):
            with torch.no_grad():
                _zn = ", ".join(
                    f"{_n}|W|={_m.weight.float().norm().item():.3e}"
                    for _n, _m in fresh_block_output_projs(fresh_block))
            print(f"[ar] fresh block zeroed-proj norms at step {step}: {_zn} "
                  f"(must grow from 0 → block is training)", flush=True)
        if _sumtok_row0 is not None:
            # NOTE: the FIRST optim.step() runs at warmup lr = 0 (LambdaLR
            # evaluates lambda(0) = 0/warmup), so Δ == 0 at step 0 is expected;
            # the row must move once lr > 0 — bf16 ULP rounding is the failure
            # mode to catch (see the fp32 value_head comments above).
            _row = sumtok_emb_weight[sumtok_id].detach().float()
            _d_new = (_row - _sumtok_row0).norm().item()
            if not _sumtok_moved[0] and _d_new > 0:
                _sumtok_moved[0] = True
                print(f"[sumtok] emb row FIRST MOVED at step {step}: "
                      f"|Δ| = {_d_new:.3e}", flush=True)
            if (step + 1) in (2, 5, 10, 20, 50, 100) or (step + 1) % 150 == 0:
                _d_nbr = (sumtok_emb_weight[sumtok_id - 1].detach().float()
                          - _sumtok_nbr0).norm().item()
                print(f"[sumtok] step {step}: cumulative |Δ emb[{sumtok_id}]| = "
                      f"{_d_new:.3e} (row norm {_row.norm().item():.3e}, init "
                      f"{_sumtok_row0.norm().item():.3e}); |Δ frozen nbr row| = "
                      f"{_d_nbr:.3e} (must be exactly 0)", flush=True)

        mean_loss = accum_loss / max(accum_n, 1)
        cur_lr = sched.get_last_lr()[0]

        log = {
            "step": step,
            "loss": mean_loss,
            "lr": cur_lr,
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "wall_s": time.time() - t0,
        }
        line = (f"step {step:04d} | loss {mean_loss:.4f} | lr {cur_lr:.2e} "
                f"| grad {log['grad_norm']:.3f} | t {log['wall_s']:.1f}s")
        if args.mode == "ar" and fve_baseline is not None:
            fve = (1.0 - mean_loss / fve_baseline) * 100.0
            log["fve_pct"] = fve
            line += f" | FVE {fve:.1f}%"
        if args.mode == "av":
            n_seen = max(1, args.batch_size * accum_n)
            log["resp_tokens"] = accum_resp_tokens
            log["mean_resp_len"] = accum_resp_tokens / n_seen
            log["ppl"] = math.exp(min(20.0, mean_loss))
            log["entropy"] = accum_av_entropy / max(accum_n, 1)  # mean response-token entropy (nats)
            line += (f" | resp_toks {accum_resp_tokens} | ppl {log['ppl']:.2f}"
                     f" | ent {log['entropy']:.3f}")
        # AR debug scalars (norms + direction match)
        if args.mode == "ar" and ar_dbg:
            log.update(ar_dbg)
            line += (f" | cos {ar_dbg['cos_pred_gold']:.3f} "
                     f"| |p|/|g| {ar_dbg['pred_norm']:.1f}/{ar_dbg['gold_norm']:.1f}")
        print(line, flush=True)

        # ---- periodic example-generation table (debug) ----
        if args.sample_every > 0 and (
            (step + 1) % args.sample_every == 0 or (step + 1) == args.num_steps
        ):
            if args.mode == "av":
                with amp():
                    samps = av_generate_samples(
                        model, tokenizer, sample_rows, cfg, device,
                        max_new_tokens=args.sample_max_new_tokens,
                        n_slots=args.n_slots,
                    )
                for s in samps:
                    sample_table_data.append([step, s["idx"],
                                              s["gen_len"], s["explanation"]])
                print(f"  [sample@{step}] {len(samps)} gens; "
                      f"e.g. idx0: {samps[0]['explanation'][:160]!r}", flush=True)
                if not args.no_wandb:
                    log["samples"] = wandb.Table(
                        columns=["step", "idx", "gen_len", "explanation"],
                        data=list(sample_table_data),
                    )
            else:  # ar: reconstruction examples
                model.eval()
                for i, row in enumerate(sample_rows):
                    ids, attn, gold = _ar_prepare_chunk(
                        [row], tokenizer, device, max_len=args.max_len,
                    )
                    with torch.no_grad(), amp():
                        pred = critic_predict(model, ids, attn, mse_scale_f)
                    pn = normalize_activation(pred, mse_scale_f)
                    gn = normalize_activation(gold, mse_scale_f)
                    rmse = F.mse_loss(pn, gn).item()
                    sample_table_data.append([
                        step, i, round(rmse, 4), row["prompt"][:400],
                    ])
                model.train()
                print(f"  [sample@{step}] logged {len(sample_rows)} AR reconstructions",
                      flush=True)
                if not args.no_wandb:
                    log["samples"] = wandb.Table(
                        columns=["step", "idx", "recon_mse", "prompt"],
                        data=list(sample_table_data),
                    )

        # ---- held-out val token-CE/ppl (AV mode, doc-disjoint) ----
        if heldout_av_rows is not None and (
            (step + 1) % args.heldout_every == 0 or (step + 1) == args.num_steps
        ):
            model.eval()
            with amp():
                h_ce, h_n = heldout_av_ce(
                    model, tokenizer, heldout_av_rows, cfg, vectors_ref, device,
                    max_len=args.max_len, n_slots=args.n_slots)
            model.train()
            log["heldout_loss"] = h_ce
            log["heldout_ppl"] = math.exp(h_ce) if h_ce < 30 else float("inf")
            print(f"  [heldout@{step}] val_loss {h_ce:.4f} | val_ppl "
                  f"{log['heldout_ppl']:.3f} (n={h_n})", flush=True)
        # ---- held-out FVE (AR mode, doc-disjoint) ----
        if heldout_pairs is not None and (
            (step + 1) % args.heldout_every == 0 or (step + 1) == args.num_steps
        ):
            # Full-FT AR sits at ~177GB steady-state (fp32 params + fp32 grads +
            # 8bit-Adam); the eval forward OOMs on the ~few-GB margin. Grads are
            # only nulled at the next step's optim.zero_grad(), so free them (and
            # the allocator cache) here to give the eval decisive headroom.
            optim.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            model.eval()
            with amp():
                h_mse, h_n = heldout_fve_mse(
                    model, tokenizer, heldout_pairs, cfg.critic_prompt_template,
                    mse_scale_f, device, max_len=args.max_len,
                )
            model.train()
            h_fve = (1.0 - h_mse / heldout_baseline) * 100.0
            log["heldout_fve_pct"] = h_fve
            log["heldout_mse"] = h_mse
            print(f"  [heldout@{step}] mse {h_mse:.4f} | FVE {h_fve:.1f}% "
                  f"(n={h_n})", flush=True)
            if args.ar_fve_by_dist and (step + 1) == args.num_steps:
                model.eval()
                with amp():
                    _byd = heldout_fve_by_dist(
                        model, tokenizer, heldout_pairs, cfg.critic_prompt_template,
                        mse_scale_f, device, heldout_baseline, max_dist=18, max_len=args.max_len)
                model.train()
                print("FVE_BY_DIST " + " ".join(f"d{d}={f:.1f}" for d, f, n in _byd), flush=True)

        if not args.no_wandb:
            wandb.log(log, step=step)

        # ---- save ----
        if (step + 1) % args.save_every == 0 or (step + 1) == args.num_steps:
            out_dir = save_dir / f"iter_{step + 1:07d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[save] → {out_dir}", flush=True)
            if args.mode == "av":
                model.save_pretrained(str(out_dir))
                tokenizer.save_pretrained(str(out_dir))
            elif args.use_lora:
                # AR + LoRA: save just the adapter weights + value_head (NOT the
                # 4-bit backbone). RL reloads via init_critic_from_base + inject.
                from safetensors.torch import save_file
                sd = {n: p.detach().cpu().contiguous()
                      for n, p in model.named_parameters()
                      if ("lora_" in n) or n.startswith("value_head")}
                if args.ar_fresh_block:
                    # The fresh block trains fully — its weights are part of
                    # the adapter checkpoint (reload = zero-init block K+1,
                    # then load these on top).
                    _fb_ids = {id(p_) for p_ in fresh_block.parameters()}
                    for n_, p_ in model.named_parameters():
                        if id(p_) in _fb_ids:
                            sd[n_] = p_.detach().cpu().contiguous()
                if sumtok_emb_weight is not None:
                    # the ONLY embedding row that trained — reload = resize +
                    # copy this row at sumtok_id (recorded in ar_meta.json).
                    sd["sumtok_embedding_row"] = (
                        sumtok_emb_weight[sumtok_id].detach().cpu().contiguous())
                save_file(sd, str(out_dir / "ar_lora_value_head.safetensors"))
                from nla.utils.arch_adapters import resolve_attn_target_modules
                (out_dir / "ar_meta.json").write_text(json.dumps({
                    "ar_num_layers": args.ar_num_layers,
                    "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                    "quant": args.quant,
                    # Resolve from the arch, not a hardcoded llama-family list:
                    # qwen3_5 adapts deltanet in_proj_qkv/in_proj_z/out_proj too,
                    # and merge_ar rebuilds the LoRA from THIS list — a wrong list
                    # makes load_state_dict see the deltanet adapters as unexpected.
                    "target_modules": resolve_attn_target_modules(model.backbone.config),
                    # Whether the backbone's final RMSNorm was stripped at init.
                    # RL must rebuild the critic the same way or predictions
                    # silently shift (pre-2026-06 ckpts: norm kept = False).
                    "final_norm_stripped": args.strip_final_norm,
                    # --ar-summary-token: id of the dedicated <|summary|> read
                    # anchor whose (sole trained) embedding row is saved as
                    # 'sumtok_embedding_row' in the safetensors. null otherwise.
                    "summary_token_id": sumtok_id,
                    # --ar-fresh-block: ar_num_layers = layer_index+2 and the
                    # last block's fully-trained weights ride along in the
                    # safetensors (reload must re-init block K+1 before load).
                    "ar_fresh_block": args.ar_fresh_block,
                }, indent=2))
                tokenizer.save_pretrained(str(out_dir))
            else:
                model.save_pretrained(str(out_dir))
                tokenizer.save_pretrained(str(out_dir))
            # Copy the sidecar so the RL trainer can find injection_token_id etc.
            import shutil
            sidecar_src = Path(args.sidecar)
            if sidecar_src.is_file() and sidecar_src.suffix == ".parquet":
                sidecar_yaml = sidecar_src.with_suffix(".parquet.nla_meta.yaml")
                if sidecar_yaml.exists():
                    shutil.copy2(sidecar_yaml, out_dir / "nla_meta.yaml")

    print("done.", flush=True)
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
