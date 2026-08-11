"""Fast, batched logprob for ONE readout file. Mean per-token logprob of each readout's first k
tokens under the base model, given the item's last-ctx-token context. Batches the rollouts per
record (one forward each), robust to short/empty readouts. Writes <out>."""
import argparse, glob, json, os
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--readout", required=True)
ap.add_argument("--evals-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--ctx", type=int, default=256)
args = ap.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)

items = {}
for f in sorted(glob.glob(os.path.join(args.evals_dir, "lens-eval-*.json"))):
    for it in json.load(open(f))["items"]:
        items[it["name"]] = it["prompt"]


def cont_ids(text):
    if not text:
        return []
    t = text if text[:1] in " \n\t.,;:!?)'\"" else " " + text
    return tok(t, add_special_tokens=False).input_ids[:args.k]


def batch_lp(ci, texts):
    conts = [cont_ids(t) for t in texts]
    conts = [c for c in conts if c]
    if not conts:
        return []
    Lc = ci.shape[1]; mk = max(len(c) for c in conts); B = len(conts)
    seq = torch.full((B, Lc + mk), tok.pad_token_id, device=dev, dtype=torch.long)
    am = torch.zeros((B, Lc + mk), device=dev, dtype=torch.long)
    seq[:, :Lc] = ci.expand(B, -1); am[:, :Lc] = 1
    for i, c in enumerate(conts):
        seq[i, Lc:Lc + len(c)] = torch.tensor(c, device=dev); am[i, Lc:Lc + len(c)] = 1
    logits = model(input_ids=seq, attention_mask=am).logits.float()
    out = []
    for i, c in enumerate(conts):
        lp = sum(torch.log_softmax(logits[i, Lc - 1 + j], -1)[tk].item() for j, tk in enumerate(c))
        out.append(lp / len(c))
    return out


recs = json.load(open(args.readout))
ctxcache, by_layer, actual_by, seen = {}, {}, {}, set()
for n, r in enumerate(recs):
    name = r["name"]
    if name not in items:
        continue
    if name not in ctxcache:
        ctxcache[name] = tok(items[name], return_tensors="pt").input_ids[:, -args.ctx:].to(dev)
    ci = ctxcache[name]
    v = batch_lp(ci, r.get("readout", []))
    if v:
        by_layer.setdefault(r["fed_layer"], []).append(float(np.mean(v)))
    if name not in seen and r.get("actual"):
        av = batch_lp(ci, r["actual"])
        if av:
            actual_by[name] = float(np.mean(av))
        seen.add(name)
    if n % 400 == 0:
        print(f"[lp] {n}/{len(recs)}", flush=True)

agg = {L: {"mean_logprob": float(np.mean(v)), "sem": float(np.std(v) / max(1, np.sqrt(len(v)))), "n": len(v)}
       for L, v in by_layer.items()}
ref = float(np.mean(list(actual_by.values()))) if actual_by else None
json.dump({"by_fed_layer": agg, "actual_reference_logprob": ref, "k": args.k, "ctx": args.ctx},
          open(args.out, "w"), indent=1)
print("actual-ref", ref, "|", " ".join(f"L{L}={agg[L]['mean_logprob']:.2f}" for L in sorted(agg, reverse=True)))
print("LP_FAST_DONE", flush=True)
