"""Objective (judge-free) faithfulness: is the lens's readout an actual model generation of the context?

For each stored fed-layer readout, compute the base model's mean per-token logprob of the readout's
first k tokens, conditioned on the item's context (last 256 tokens = the window the activation was
grabbed from). High = the readout is a plausible model continuation; low = it isn't (e.g. workspace
content, or garbage). Also scores the model's OWN actual continuation as a reference ceiling.
Reuses the already-generated readouts. Writes <readouts>_logprob.json per file.
"""
import argparse, glob, json, os
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--readouts", nargs="+", required=True)
ap.add_argument("--evals-dir", required=True)
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--k", type=int, default=3)         # first k readout tokens
ap.add_argument("--ctx", type=int, default=256)     # last-N-token context window
args = ap.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
model = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)

items = {}
for f in sorted(glob.glob(os.path.join(args.evals_dir, "lens-eval-*.json"))):
    for it in json.load(open(f))["items"]:
        items[it["name"]] = it["prompt"]


def logprob(ctx_ids, text, k):
    """mean per-token logprob of the first k tokens of `text` continuing ctx_ids."""
    if not text:
        return None
    t = text if text[:1] in " \n\t.,;:!?)'\"" else " " + text     # continuations usually lead with a space
    ri = tok(t, add_special_tokens=False).input_ids[:k]
    if not ri:
        return None
    full = torch.cat([ctx_ids, torch.tensor([ri], device=dev)], 1)
    logits = model(full).logits[0].float()
    Lc = ctx_ids.shape[1]
    tot = 0.0
    for i, tkn in enumerate(ri):
        tot += torch.log_softmax(logits[Lc - 1 + i], -1)[tkn].item()
    return tot / len(ri)


for rf in args.readouts:
    recs = json.load(open(rf))
    ctxcache, by_layer, actual_by = {}, {}, {}
    seen_actual = set()
    for j, r in enumerate(recs):
        name, L = r["name"], r["fed_layer"]
        if name not in items:
            continue
        if name not in ctxcache:
            ci = tok(items[name], return_tensors="pt").input_ids[:, -args.ctx:].to(dev)
            ctxcache[name] = ci
        ci = ctxcache[name]
        vals = [logprob(ci, ro, args.k) for ro in r.get("readout", [])]
        vals = [v for v in vals if v is not None]
        if vals:
            by_layer.setdefault(L, []).append(float(np.mean(vals)))
        # actual-continuation reference (per item, once)
        if name not in seen_actual and r.get("actual"):
            av = [logprob(ci, a, args.k) for a in r["actual"]]
            av = [v for v in av if v is not None]
            if av:
                actual_by[name] = float(np.mean(av))
            seen_actual.add(name)
        if j % 400 == 0:
            print(f"[lp] {os.path.basename(rf)} rec {j}/{len(recs)}", flush=True)
    agg = {L: {"mean_logprob": float(np.mean(v)), "sem": float(np.std(v) / max(1, np.sqrt(len(v)))), "n": len(v)}
           for L, v in by_layer.items()}
    ref = float(np.mean(list(actual_by.values()))) if actual_by else None
    out = {"by_fed_layer": agg, "actual_reference_logprob": ref, "k": args.k, "ctx": args.ctx}
    op = rf.replace(".json", "_logprob.json")
    json.dump(out, open(op, "w"), indent=1)
    print(f"[lp] wrote {os.path.basename(op)} | actual-ref={ref:.3f} | " +
          " ".join(f"L{L}={agg[L]['mean_logprob']:.2f}" for L in sorted(agg, reverse=True)), flush=True)
print("LOGPROB_DONE", flush=True)
