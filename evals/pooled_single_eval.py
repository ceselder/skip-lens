"""Arm I readout: feed the AVERAGED-gradient pooled vector to the single-slot AV.

    test vector = [ sum_d Jbar^(d) ] @ h42[p]        (the paper's J-lens vector)

The arm was trained on the same operator shape without the expectation over
contexts, so this eval changes exactly one thing: averaged vs per-example
gradient. Output rows are judge_fedlayer.py-compatible.

Also runs the two controls that matter for a single-vector lens:
    raw_h42     feed h42 itself (does the transport earn its keep? measured
                cos(raw h42, real h62) = 0.52 vs 0.585 for the transport, so
                this control is closer than one might hope)
    mean_only   feed the corpus-mean transported vector, identical for every
                position — the floor any real readout must clear, since
                cos(real state, mu_62) is already ~0.5
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
from pretrain.build_pooled_single import SINGLE_TEMPLATE

ap = argparse.ArgumentParser()
ap.add_argument("--av-ckpt", required=True)
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--src-layer", type=int, default=42)
ap.add_argument("--tgt-layer", type=int, default=62)
ap.add_argument("--n-offsets", type=int, default=16)
ap.add_argument("--evals-dir", required=True)
ap.add_argument("--conditions", default="pooled_avg,raw_h42,mean_only")
ap.add_argument("--n-ao", type=int, default=4)
ap.add_argument("--rollout-len", type=int, default=24)
ap.add_argument("--topk", type=int, default=12)
ap.add_argument("--max-items", type=int, default=999)
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda"
CONDS = args.conditions.split(",")

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
base = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16,
    attn_implementation="sdpa").to(dev).eval()
model = PeftModel.from_pretrained(base, args.av_ckpt, adapter_name="ps").eval()
inj_char, inj_id = find_injection_token(tok)
left, right = compute_canonical_neighbors(tok, SINGLE_TEMPLATE, inj_char, inj_id)
vref = [None]
register_karvonen_hook(model, vref, inj_id, left, right)
torch.set_grad_enabled(False)

# token-pooled AVERAGED gradient = sum over offsets of the averaged per-offset J
JP = None
for d in range(args.n_offsets):
    p = os.path.join(args.jbar_dir,
                     f"Jbar_L{args.src_layer}_to_L{args.tgt_layer}_off{d}.npy")
    if not os.path.exists(p):
        break
    M = torch.from_numpy(np.load(p)).float().to(dev)
    JP = M if JP is None else JP + M
assert JP is not None, f"no Jbar matrices in {args.jbar_dir}"
HBAR = torch.from_numpy(np.load(
    os.path.join(args.jbar_dir, f"hbar_L{args.src_layer}.npy"))).float().to(dev)
MEANVEC = JP @ HBAR
print(f"[ps] pooled averaged gradient loaded (|JP|={float(JP.norm()):.1f}, "
      f"|JP.hbar|={float(MEANVEC.norm()):.1f})", flush=True)

grab = {}
base_causal(model).model.layers[args.src_layer].register_forward_hook(
    lambda m, i, o: grab.__setitem__(
        args.src_layer, (o[0] if isinstance(o, tuple) else o).detach()))

GEN_PAD = 256
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
_pc = {}


def prompt_ids():
    if "i" not in _pc:
        s = tok.apply_chat_template(
            [{"role": "user", "content": SINGLE_TEMPLATE.format(injection_char=inj_char)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        _pc["i"] = torch.tensor([tok.encode(s, add_special_tokens=False)], device=dev)
    return _pc["i"]


def pad_for(ids, side):
    real = ids[:, -GEN_PAD:]
    n = real.shape[1]
    k = GEN_PAD - n
    if k <= 0:
        return real, torch.ones_like(real), n - 1
    pad = torch.full((1, k), PAD, device=dev, dtype=real.dtype)
    ones = torch.ones(1, n, dtype=torch.long, device=dev)
    zeros = torch.zeros(1, k, dtype=torch.long, device=dev)
    if side == "right":
        return torch.cat([real, pad], 1), torch.cat([ones, zeros], 1), n - 1
    return torch.cat([pad, real], 1), torch.cat([zeros, ones], 1), n - 1


def vec_for(cond, h42):
    if cond == "pooled_avg":
        return JP @ h42
    if cond == "raw_h42":
        return h42
    if cond == "mean_only":
        return MEANVEC
    raise ValueError(cond)


def roll(v, n, mx, temp=0.7):
    pt = prompt_ids()
    ids = pt.repeat(max(1, n), 1)
    vref[0] = v.float().unsqueeze(0).repeat(max(1, n), 1).contiguous()
    try:
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                           max_new_tokens=mx, do_sample=(temp > 0),
                           temperature=max(temp, 1e-5), top_p=0.95,
                           pad_token_id=tok.eos_token_id)
    finally:
        vref[0] = None
    return [tok.decode(x[pt.shape[1]:], skip_special_tokens=True).strip() for x in g]


items = []
for f in sorted(glob.glob(os.path.join(args.evals_dir, "lens-eval-*.json"))):
    for it in json.load(open(f))["items"][: args.max_items]:
        items.append({"name": it["name"], "prompt": it["prompt"]})
print(f"[ps] {len(items)} items x {len(CONDS)} conditions", flush=True)

out = []
for j, it in enumerate(items):
    ids = tok(it["prompt"], return_tensors="pt").input_ids.to(dev)
    gids, gmask, last = pad_for(ids, "right")
    pids, pmask, _ = pad_for(ids, "left")
    with model.disable_adapter():
        model(input_ids=gids, attention_mask=gmask)
        h42 = grab[args.src_layer][0, last].float()
        g = model.generate(input_ids=pids.repeat(4, 1), attention_mask=pmask.repeat(4, 1),
                           do_sample=True, temperature=0.8, top_p=0.95,
                           max_new_tokens=args.rollout_len, pad_token_id=tok.eos_token_id)
        actual = [tok.decode(x[pids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    ji, _ = lens_topk(model, (JP @ h42).float(), k=args.topk)
    jl_top = [tok.decode([int(i)]) for i in ji]
    ctx = it["prompt"][-200:]
    model.set_adapter("ps")
    for cond in CONDS:
        out.append({"name": it["name"], "fed_layer": args.src_layer,
                    "condition": cond, "jlens_top": jl_top, "actual": actual,
                    "context": ctx,
                    "readout": roll(vec_for(cond, h42), args.n_ao, args.rollout_len)})
    if j % 25 == 0:
        print(f"[ps] item {j}/{len(items)}", flush=True)

json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
print(f"[ps] wrote {args.out} ({len(out)} records)", flush=True)
