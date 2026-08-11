"""Diagnostic: separate the AV's held-out generalization CE from my scaling-metric confounds.

For held-out (disjoint) docs, compute three next-token CEs of the AV (nats):
  A) CE_stored+labeled : inject the STORED full-context L62 activation, teacher-force the
                         LABELED rollout (this matches the training-loss computation, but on
                         held-out docs) -> the honest generalization number.
  B) CE_recap+labeled  : inject the activation RE-captured from the 48-tok ctx_text tail,
                         teacher-force the labeled rollout -> isolates the recapture/context loss.
  C) CE_recap+temp1    : recaptured activation + a FRESH temp-1 sample (== the scaling-curve metric).

Run on an early + a late checkpoint: if A drops with training (tracking train loss 2.46->1.98)
the AV generalizes and the flat 4.5 curve was A->C confounds; if A is flat ~4.5 it's overfitting.
"""
import argparse
import numpy as np
import torch
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nla.utils.hooks import register_karvonen_hook
from nla.schema import compute_canonical_neighbors
from nla.datagen.injection_tokens import find_injection_token
from nla.utils.arch_adapters import resolve_text_model

ACTOR_TEMPLATE = (
    "You are shown an internal activation vector captured from a language model "
    "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
    "taken at one position and encodes what the model is about to generate next. "
    "Output the text the model most likely produces immediately after this point.\n\n"
    "<concept>{injection_char}</concept>")

ap = argparse.ArgumentParser()
ap.add_argument("--av-ckpt", required=True)
ap.add_argument("--ctx-parquet", required=True)
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--tag", default="")
args = ap.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
model = PeftModel.from_pretrained(base, args.av_ckpt, adapter_name="AV").eval()
inj_char, inj_id = find_injection_token(tok)
left, right = compute_canonical_neighbors(tok, ACTOR_TEMPLATE, inj_char, inj_id)
vref = [None]
register_karvonen_hook(model, vref, inj_id, left, right)
model._fl_vref = vref
torch.set_grad_enabled(False)

inner = model.get_base_model() if hasattr(model, "get_base_model") else model
layers = resolve_text_model(inner).model.layers
grab = {}
layers[62].register_forward_hook(
    lambda m, i, o: grab.__setitem__(62, (o[0] if isinstance(o, tuple) else o).detach()))


def prompt_ids():
    content = ACTOR_TEMPLATE.format(injection_char=inj_char)
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return torch.tensor([tok.encode(s, add_special_tokens=False)], device=dev)


def _trunc_eos(ids):
    nz = (ids == tok.eos_token_id).nonzero()
    return ids[: int(nz[0])] if nz.numel() else ids


@torch.no_grad()
def ce(h_l, cont_ids):
    if cont_ids.numel() == 0:
        return float("nan")
    pt = prompt_ids()
    seq = torch.cat([pt, cont_ids.view(1, -1)], 1)
    model._fl_vref[0] = h_l.view(1, -1)
    try:
        logits = model(input_ids=seq, attention_mask=torch.ones_like(seq)).logits[0].float()
    finally:
        model._fl_vref[0] = None
    P = pt.shape[1]
    pred = logits[P - 1: P - 1 + cont_ids.shape[0]]
    lp = pred.log_softmax(-1).gather(-1, cont_ids.view(-1, 1)).squeeze(-1)
    return float(-lp.mean())


@torch.no_grad()
def recap(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev)
    with model.disable_adapter():
        model(input_ids=ids)
    return grab[62][0, ids.shape[1] - 1].float().clone(), ids


t = pq.read_table(args.ctx_parquet)
acts = t.column("activation_vector").to_pylist()
rolls = t.column("rollout_token_ids").to_pylist()
ctxs = t.column("ctx_text").to_pylist()
A, B, C = [], [], []
torch.manual_seed(0)
for i in range(min(args.n, len(ctxs))):
    if not rolls[i] or not ctxs[i]:
        continue
    roll = _trunc_eos(torch.tensor(rolls[i][0], dtype=torch.long, device=dev))
    if roll.numel() == 0:
        continue
    stored = torch.tensor(acts[i], dtype=torch.float32, device=dev)
    A.append(ce(stored, roll))                              # stored act + labeled rollout
    hrec, ids = recap(ctxs[i])
    B.append(ce(hrec, roll))                                # recap act + labeled rollout
    with model.disable_adapter():
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                           max_new_tokens=24, do_sample=True, temperature=1.0, top_p=1.0,
                           pad_token_id=tok.eos_token_id)
    samp = _trunc_eos(g[0, ids.shape[1]:])
    C.append(ce(hrec, samp) if samp.numel() else float("nan"))

import math
mean = lambda xs: float(np.mean([x for x in xs if math.isfinite(x)]))
print(f"[{args.tag}] n={len(A)} | A CE_stored+labeled={mean(A):.3f} | "
      f"B CE_recap+labeled={mean(B):.3f} | C CE_recap+temp1(scaling metric)={mean(C):.3f}", flush=True)
