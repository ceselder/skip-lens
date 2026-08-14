"""FED-LAYER SWEEP for the mean-subtraction ablation.

Feed a futurelens the residual activation from a range of depths on the DISAGREEMENT set, and
record its readout tagged by fed-layer. Two knobs vs the base sweep:
  * LOCAL per-layer J-lens reference: agree_jlens is scored against each fed layer's OWN workspace,
    jl_top[l] = topk(lens(J_{l->62} . h_l))  (identity at 62). This is the fair per-layer scoring.
  * --sub-mean-npz: subtract a per-layer mean from the fed activation before injecting
    (feed h_l - mean_l). For the mean-subtraction lens, pass the TEST-distribution per-layer means;
    for the raw control, omit it (feed raw h_l).
The J-lens reference is always computed on the RAW h_l (the workspace is a property of the model's
real computation, not of the injected representation).
"""
import argparse
import glob
import json
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nla.utils.hooks import register_karvonen_hook
from nla.schema import compute_canonical_neighbors
from nla.datagen.injection_tokens import find_injection_token
from fl_common import lens_topk, base_causal, ACTOR_TEMPLATE, _prompt_ids

ap = argparse.ArgumentParser()
ap.add_argument("--ao-ckpt", required=True)
ap.add_argument("--jsnap-dir", required=True, help="dir with per-layer J_L{l}_to_L{jtarget}.npy")
ap.add_argument("--jtarget", type=int, default=62)
ap.add_argument("--fed-layers", default="62,55,48,42,34,26")
ap.add_argument("--sub-mean-npz", default=None, help="npz of per-layer means to subtract from fed h_l")
ap.add_argument("--normalize-first", action="store_true", help="feed h_l/||h_l|| - mean_dir_l (mean taken in unit space, magnitude-free) instead of raw h_l - mean_l")
ap.add_argument("--whiten-npz", default=None, help="npz of per-layer ZCA stats (mu_l, W_l); feed W_l @ (h_l - mu_l)")
ap.add_argument("--chat-template", action="store_true", help="wrap each probe as a user turn + add_generation_prompt, harvest at the assistant anchor (chat-native), instead of raw text last-token")
ap.add_argument("--evals-dir", required=True)
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--n-ao", type=int, default=4)
ap.add_argument("--rollout-len", type=int, default=16)
ap.add_argument("--topk", type=int, default=12)
ap.add_argument("--max-items", type=int, default=999)
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda"
FED = [int(x) for x in args.fed_layers.split(",")]
GRAB_LAYERS = sorted(set(FED))

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
base = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
model = PeftModel.from_pretrained(base, args.ao_ckpt).eval()
inj_char, inj_id = find_injection_token(tok)
left, right = compute_canonical_neighbors(tok, ACTOR_TEMPLATE, inj_char, inj_id)
vref = [None]
register_karvonen_hook(model, vref, inj_id, left, right)
model._fl_vref = vref
torch.set_grad_enabled(False)

Jloc = {}                                         # per-layer J_{l->jtarget} for the LOCAL workspace reference
for l in FED:
    if l == args.jtarget:
        continue
    p = os.path.join(args.jsnap_dir, f"J_L{l}_to_L{args.jtarget}.npy")
    Jloc[l] = torch.from_numpy(np.load(p)).float().to(dev)
submean = {}
if args.sub_mean_npz:
    z = np.load(args.sub_mean_npz)
    for l in FED:
        if str(l) in z:
            submean[l] = torch.from_numpy(z[str(l)]).float().to(dev)
    print(f"[fed] subtracting per-layer test means for layers {sorted(submean)}", flush=True)
whiten = {}
if args.whiten_npz:
    z = np.load(args.whiten_npz)
    for l in FED:
        if f"W_{l}" in z:
            whiten[l] = (torch.from_numpy(z[f"mu_{l}"]).float().to(dev),
                         torch.from_numpy(z[f"W_{l}"]).float().to(dev))
    print(f"[fed] ZCA-whitening fed activation for layers {sorted(whiten)}", flush=True)

grab = {}
for l in GRAB_LAYERS:
    base_causal(model).model.layers[l].register_forward_hook(
        lambda m, i, o, L=l: grab.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))

GEN_PAD = 256
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def grab_inputs(ids):
    real = ids[:, -GEN_PAD:]; n = real.shape[1]; rp = GEN_PAD - n
    if rp <= 0:
        return real, torch.ones_like(real), n - 1
    pad = torch.full((1, rp), PAD_ID, device=real.device, dtype=real.dtype)
    mask = torch.cat([torch.ones(1, n, dtype=torch.long, device=real.device),
                      torch.zeros(1, rp, dtype=torch.long, device=real.device)], 1)
    return torch.cat([real, pad], 1), mask, n - 1


def gen_inputs(ids):
    real = ids[:, -GEN_PAD:]; n = real.shape[1]; lp = GEN_PAD - n
    if lp <= 0:
        return real, torch.ones_like(real)
    pad = torch.full((1, lp), PAD_ID, device=real.device, dtype=real.dtype)
    mask = torch.cat([torch.zeros(1, lp, dtype=torch.long, device=real.device),
                      torch.ones(1, n, dtype=torch.long, device=real.device)], 1)
    return torch.cat([pad, real], 1), mask


def brollout(activation, n, max_new, temp=0.7):
    act = torch.as_tensor(activation, dtype=torch.float32, device=dev).view(1, -1)
    pt = _prompt_ids(tok, dev); B = max(1, n)
    ids = pt.repeat(B, 1)
    model._fl_vref[0] = act.expand(B, -1).contiguous()
    try:
        g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                           do_sample=(temp > 0), temperature=max(temp, 1e-5), top_p=0.95,
                           pad_token_id=tok.eos_token_id)
    finally:
        model._fl_vref[0] = None
    return [tok.decode(x[pt.shape[1]:], skip_special_tokens=True).strip() for x in g]


items = []
for f in sorted(glob.glob(os.path.join(args.evals_dir, "lens-eval-*.json"))):
    for it in json.load(open(f))["items"][: args.max_items]:
        items.append({"name": it["name"], "prompt": it["prompt"]})
print(f"[fed] {len(items)} disagreement items x {len(FED)} fed-layers {FED}", flush=True)

def prompt_ids(text):
    if args.chat_template:                       # chat-native: user turn + assistant anchor
        s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
        return tok(s, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    return tok(text, return_tensors="pt").input_ids.to(dev)


out = []
for j, it in enumerate(items):
    ids = prompt_ids(it["prompt"])
    gids, gmask, last = grab_inputs(ids)
    pids, pmask = gen_inputs(ids)
    with model.disable_adapter():
        model(input_ids=gids, attention_mask=gmask)
        h = {l: grab[l][0, last].float() for l in GRAB_LAYERS}
        g = model.generate(input_ids=pids.repeat(4, 1), attention_mask=pmask.repeat(4, 1),
                           do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=24,
                           pad_token_id=tok.eos_token_id)
        actual = [tok.decode(x[pids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    ctx = it["prompt"][-200:]
    for l in FED:
        Jh_l = h[l] if l == args.jtarget else (Jloc[l] @ h[l]).float()   # LOCAL workspace on RAW h_l
        ji, _ = lens_topk(model, Jh_l, k=args.topk)
        jl_top = [tok.decode([int(i)]) for i in ji]
        if l in whiten:                                                      # ZCA: feed W_l (h_l - mu_l)
            mu, W = whiten[l]
            fed_vec = W @ (h[l] - mu)
        else:
            hl = h[l] / (h[l].norm() + 1e-8) if args.normalize_first else h[l]   # unit-space if normed ablation
            fed_vec = hl - submean[l] if l in submean else hl                    # deviation-from-mean (or raw)
        out.append({"name": it["name"], "fed_layer": l, "jlens_top": jl_top,
                    "actual": actual, "context": ctx,
                    "readout": brollout(fed_vec, args.n_ao, args.rollout_len)})
    if j % 10 == 0:
        print(f"[fed] item {j}/{len(items)}", flush=True)

json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
print(f"[fed] wrote {args.out} ({len(out)} records = {len(items)} items x {len(FED)} fed-layers)", flush=True)
