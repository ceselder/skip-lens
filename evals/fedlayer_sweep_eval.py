"""FED-LAYER SWEEP: feed the penultimate (L62) claude workspace-AO the RAW residual activation
from a range of depths, on the DISAGREEMENT set (J-lens argmax != final answer), and record its
readout tagged by fed-layer, alongside the (fixed) J-lens top-k and the model's actual continuations.

The judge (judge_fedlayer.py) then scores each readout twice: agreement with the J-lens top-k
(workspace proxy) vs agreement with the actual continuation (the answer). Discriminates a faithful
readout of the intermediate state (tracks J-lens, degrades with depth) from a tuned-lens that
decodes the answer regardless of fed depth. No Jacobian on the fed activation — raw h_l injected."""
import argparse, glob, json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nla.utils.hooks import register_karvonen_hook
from nla.schema import compute_canonical_neighbors
from nla.datagen.injection_tokens import find_injection_token
from fl_common import lens_topk, base_causal, ACTOR_TEMPLATE, _prompt_ids

ap = argparse.ArgumentParser()
ap.add_argument("--ao-ckpt", required=True)              # ao_v4_L62/iter_0001000 (penultimate claude AO)
ap.add_argument("--jdir", required=True)                 # holds J_L{jorigin}_to_L{jtarget}.npy
ap.add_argument("--jorigin", type=int, default=42)
ap.add_argument("--jtarget", type=int, default=62)
ap.add_argument("--fed-layers", default="62,55,48,42,34,26,18,10")
ap.add_argument("--evals-dir", required=True)
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--n-ao", type=int, default=4)
ap.add_argument("--rollout-len", type=int, default=24)
ap.add_argument("--topk", type=int, default=12)
ap.add_argument("--max-items", type=int, default=999)
ap.add_argument("--out", required=True)
ap.add_argument("--out-jac", default=None, help="if set, also feed J_{l->62}.h_l (per-layer Jacobian) and write jac-arm readouts here")
args = ap.parse_args()
dev = "cuda"
FED = [int(x) for x in args.fed_layers.split(",")]
GRAB_LAYERS = sorted(set(FED + [args.jorigin]))

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
base = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
model = PeftModel.from_pretrained(base, args.ao_ckpt, adapter_name="L62").eval()
inj_char, inj_id = find_injection_token(tok)
left, right = compute_canonical_neighbors(tok, ACTOR_TEMPLATE, inj_char, inj_id)
vref = [None]; register_karvonen_hook(model, vref, inj_id, left, right); model._fl_vref = vref
torch.set_grad_enabled(False)

J = torch.from_numpy(np.load(os.path.join(args.jdir, f"J_L{args.jorigin}_to_L{args.jtarget}.npy"))).float().to(dev)
Jmap = {}                                                # per-layer J_{l->jtarget} for the jac arm
if args.out_jac:
    for l in FED:
        p = os.path.join(args.jdir, f"J_L{l}_to_L{args.jtarget}.npy")
        if os.path.exists(p):
            Jmap[l] = torch.from_numpy(np.load(p)).float().to(dev)
    print(f"[fed] jac arm ON: J_{{l->{args.jtarget}}} available for layers {sorted(Jmap)}", flush=True)
grab = {}
for l in GRAB_LAYERS:                                    # grab every fed layer's residual in one forward
    base_causal(model).model.layers[l].register_forward_hook(
        lambda m, i, o, L=l: grab.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))

GEN_PAD = 256
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def grab_inputs(ids):                                    # right-pad, grab last-real-token index (exact)
    real = ids[:, -GEN_PAD:]; n = real.shape[1]; rp = GEN_PAD - n
    if rp <= 0:
        return real, torch.ones_like(real), n - 1
    pad = torch.full((1, rp), PAD_ID, device=real.device, dtype=real.dtype)
    mask = torch.cat([torch.ones(1, n, dtype=torch.long, device=real.device),
                      torch.zeros(1, rp, dtype=torch.long, device=real.device)], 1)
    return torch.cat([real, pad], 1), mask, n - 1


def gen_inputs(ids):                                     # left-pad for batched generation
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

out = []
out_jac = []
for j, it in enumerate(items):
    ids = tok(it["prompt"], return_tensors="pt").input_ids.to(dev)
    gids, gmask, last = grab_inputs(ids)
    pids, pmask = gen_inputs(ids)
    with model.disable_adapter():
        model(input_ids=gids, attention_mask=gmask)               # populate grab for all layers
        h = {l: grab[l][0, last].float() for l in GRAB_LAYERS}     # last-real-token activation per layer
        g = model.generate(input_ids=pids.repeat(4, 1), attention_mask=pmask.repeat(4, 1),
                           do_sample=True, temperature=0.8, top_p=0.95, max_new_tokens=args.rollout_len,
                           pad_token_id=tok.eos_token_id)
        actual = [tok.decode(x[pids.shape[1]:], skip_special_tokens=True).strip() for x in g]
    Jh = (J @ h[args.jorigin]).float()                             # FIXED J-lens reference (from jorigin)
    ji, _ = lens_topk(model, Jh, k=args.topk)
    jl_top = [tok.decode([int(i)]) for i in ji]
    ctx = it["prompt"][-200:]
    model.set_adapter("L62")
    for l in FED:                                                  # feed RAW h_l (no Jacobian) to the L62 AO
        out.append({"name": it["name"], "fed_layer": l, "jlens_top": jl_top,
                    "actual": actual, "context": ctx,
                    "readout": brollout(h[l], args.n_ao, args.rollout_len)})
    for l in FED:                                                  # jac arm: feed J_{l->62}.h_l (mapped into penultimate space)
        if l in Jmap:
            Jh_l = (Jmap[l] @ h[l]).float()
            out_jac.append({"name": it["name"], "fed_layer": l, "jlens_top": jl_top,
                            "actual": actual, "context": ctx,
                            "readout": brollout(Jh_l, args.n_ao, args.rollout_len)})
    if j % 5 == 0:
        print(f"[fed] item {j}/{len(items)}", flush=True)

json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
print(f"[fed] wrote {args.out} ({len(out)} records = {len(items)} items x {len(FED)} fed-layers)", flush=True)
if args.out_jac:
    json.dump(out_jac, open(args.out_jac, "w"), ensure_ascii=False, indent=1)
    print(f"[fed] wrote {args.out_jac} ({len(out_jac)} jac records, layers {sorted(Jmap)})", flush=True)
