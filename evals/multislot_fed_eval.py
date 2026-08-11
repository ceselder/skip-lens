"""MULTI-SLOT FED EVAL: feed the K-slot AV the averaged per-offset transports
[Jbar^(0) h42, ..., Jbar^(K-1) h42] on the eval items, and record readouts.

This is the arm-A test protocol: training saw LOCAL-Jacobian transports; here
the slots are the corpus-AVERAGED transports (the deliberate mismatch that
projects onto broadly-verbalizable content). Conditions:

  per_offset        the design: slot d = Jbar^(d) @ h42
  pooled_identical  K copies of Jbar_offpooled @ h42 (does horizon structure
                    matter, or just slot count?)
  slot0_only        slot 0 kept, other slots zeroed (is the AV reading late
                    slots at all, or autocompleting off slot 0?)
  no_slot0          slot 0 zeroed, slots 1..K-1 kept
  shuffled_slots    per-offset vectors in a fixed derangement of slot order

Output rows are judge_fedlayer.py-compatible ({name, fed_layer, jlens_top,
actual, context, readout}) with an extra "condition" field; the fixed
reference lens is Jbar_offpooled (same family as the fed transports).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from fl_common import base_causal, lens_topk
from nla.datagen.injection_tokens import find_injection_token
from nla.schema import compute_canonical_neighbors
from nla.utils.hooks import register_karvonen_hook

ACTOR_TEMPLATE_MULTI = (
    "You are shown {k} internal activation vectors captured from a language "
    "model as it reads a passage of text. The vectors, enclosed in <concept> "
    "tags, are the model's state at one position transported to {k} "
    "consecutive future positions: the first vector encodes what the model is "
    "about to generate next, the second what it will generate after that, and "
    "so on. Output the text the model most likely produces over these {k} "
    "positions.\n\n<concept>{markers}</concept>")

ap = argparse.ArgumentParser()
ap.add_argument("--av-ckpt", required=True, help="multi-slot (arm A) adapter dir")
ap.add_argument("--jbar-dir", required=True,
                help="holds Jbar_L{src}_to_L{tgt}_off{d}.npy + _offpooled.npy")
ap.add_argument("--src-layer", type=int, default=42)
ap.add_argument("--tgt-layer", type=int, default=62)
ap.add_argument("--k-slots", type=int, default=8)
ap.add_argument("--evals-dir", required=True)
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--conditions",
                default="per_offset,pooled_identical,slot0_only,no_slot0")
ap.add_argument("--n-ao", type=int, default=4)
ap.add_argument("--rollout-len", type=int, default=24)
ap.add_argument("--topk", type=int, default=12)
ap.add_argument("--max-items", type=int, default=999)
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda"
K = args.k_slots
CONDS = args.conditions.split(",")

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
base = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16,
    attn_implementation="sdpa").to(dev).eval()
model = PeftModel.from_pretrained(base, args.av_ckpt, adapter_name="msA").eval()
inj_char, inj_id = find_injection_token(tok)
TEMPLATE = ACTOR_TEMPLATE_MULTI.format(k=K, markers="{injection_char}" * K)
left, right = compute_canonical_neighbors(tok, TEMPLATE, inj_char, inj_id)
vref = [None]
register_karvonen_hook(model, vref, inj_id, left, right)
torch.set_grad_enabled(False)

Jbar = {}
for d in range(K):
    p = os.path.join(args.jbar_dir,
                     f"Jbar_L{args.src_layer}_to_L{args.tgt_layer}_off{d}.npy")
    Jbar[d] = torch.from_numpy(np.load(p)).float().to(dev)
Jpool = torch.from_numpy(np.load(os.path.join(
    args.jbar_dir,
    f"Jbar_L{args.src_layer}_to_L{args.tgt_layer}_offpooled.npy"))).float().to(dev)
print(f"[ms] loaded {len(Jbar)} per-offset Jbar + pooled", flush=True)

grab = {}
base_causal(model).model.layers[args.src_layer].register_forward_hook(
    lambda m, i, o: grab.__setitem__(
        args.src_layer, (o[0] if isinstance(o, tuple) else o).detach()))

GEN_PAD = 256
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

_prompt_cache = {}


def prompt_ids():
    if "ids" not in _prompt_cache:
        content = TEMPLATE.format(injection_char=inj_char)
        s = tok.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        _prompt_cache["ids"] = torch.tensor(
            [tok.encode(s, add_special_tokens=False)], dtype=torch.long,
            device=dev)
    return _prompt_cache["ids"]


def grab_inputs(ids):
    real = ids[:, -GEN_PAD:]
    n = real.shape[1]
    rp = GEN_PAD - n
    if rp <= 0:
        return real, torch.ones_like(real), n - 1
    pad = torch.full((1, rp), PAD_ID, device=real.device, dtype=real.dtype)
    mask = torch.cat([torch.ones(1, n, dtype=torch.long, device=real.device),
                      torch.zeros(1, rp, dtype=torch.long, device=real.device)], 1)
    return torch.cat([real, pad], 1), mask, n - 1


def gen_inputs(ids):
    real = ids[:, -GEN_PAD:]
    n = real.shape[1]
    lp = GEN_PAD - n
    if lp <= 0:
        return real, torch.ones_like(real)
    pad = torch.full((1, lp), PAD_ID, device=real.device, dtype=real.dtype)
    mask = torch.cat([torch.zeros(1, lp, dtype=torch.long, device=real.device),
                      torch.ones(1, n, dtype=torch.long, device=real.device)], 1)
    return torch.cat([pad, real], 1), mask


def slots_for(cond, h42):
    per = torch.stack([Jbar[d] @ h42 for d in range(K)])  # [K, d]
    if cond == "per_offset":
        return per
    if cond == "pooled_identical":
        return (Jpool @ h42).expand(K, -1).contiguous()
    if cond == "slot0_only":
        s = torch.zeros_like(per)
        s[0] = per[0]
        return s
    if cond == "no_slot0":
        s = per.clone()
        s[0] = 0
        return s
    if cond == "shuffled_slots":
        perm = (torch.arange(K) + K // 2) % K
        return per[perm]
    raise ValueError(cond)


def brollout_slots(slots, n, max_new, temp=0.7):
    pt = prompt_ids()
    B = max(1, n)
    ids = pt.repeat(B, 1)
    vref[0] = slots.float().repeat(B, 1).contiguous()  # [B*K, d], row-major
    try:
        g = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new, do_sample=(temp > 0),
            temperature=max(temp, 1e-5), top_p=0.95,
            pad_token_id=tok.eos_token_id)
    finally:
        vref[0] = None
    return [tok.decode(x[pt.shape[1]:], skip_special_tokens=True).strip()
            for x in g]


items = []
for f in sorted(glob.glob(os.path.join(args.evals_dir, "lens-eval-*.json"))):
    for it in json.load(open(f))["items"][: args.max_items]:
        items.append({"name": it["name"], "prompt": it["prompt"]})
print(f"[ms] {len(items)} items x {len(CONDS)} conditions", flush=True)

out = []
for j, it in enumerate(items):
    ids = tok(it["prompt"], return_tensors="pt").input_ids.to(dev)
    gids, gmask, last = grab_inputs(ids)
    pids, pmask = gen_inputs(ids)
    with model.disable_adapter():
        model(input_ids=gids, attention_mask=gmask)
        h42 = grab[args.src_layer][0, last].float()
        g = model.generate(
            input_ids=pids.repeat(4, 1), attention_mask=pmask.repeat(4, 1),
            do_sample=True, temperature=0.8, top_p=0.95,
            max_new_tokens=args.rollout_len, pad_token_id=tok.eos_token_id)
        actual = [tok.decode(x[pids.shape[1]:], skip_special_tokens=True).strip()
                  for x in g]
    Jh = (Jpool @ h42).float()
    ji, _ = lens_topk(model, Jh, k=args.topk)
    jl_top = [tok.decode([int(i)]) for i in ji]
    ctx = it["prompt"][-200:]
    model.set_adapter("msA")
    for cond in CONDS:
        out.append({
            "name": it["name"], "fed_layer": args.src_layer,
            "condition": cond, "jlens_top": jl_top, "actual": actual,
            "context": ctx,
            "readout": brollout_slots(slots_for(cond, h42), args.n_ao,
                                      args.rollout_len),
        })
    if j % 5 == 0:
        print(f"[ms] item {j}/{len(items)}", flush=True)

json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
print(f"[ms] wrote {args.out} ({len(out)} records)", flush=True)
