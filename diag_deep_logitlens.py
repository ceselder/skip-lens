"""Does the deep-pooled AVERAGED Jacobian decode to the actual future tokens?

Every measurement so far has been vector-space cosine against the local
transport, and by that measure the deep offsets are hopeless (~0.15). But cosine
in a 5120-dim space is a harsh test and it is not the test the J-lens itself
passes: the paper reads its averaged Jacobian through the unembedding, which is a
forgiving readout that only needs the right directions to win a top-k contest.

So ask the question the way the lens actually asks it. Take real on-policy rows
(context, plus the model's own next 16 sampled tokens) and decode each candidate
vector through the unembedding:

    raw h42          the control — how much future does the activation alone show
    Jbar^(0) h42     the paper's J-lens vector (current-token dominated)
    deep pooled      Σ_{d≥1} w_d Jbar^(d) h42, norm-equalised — the vector whose
                     cosine says it is useless but whose CONTENT is the question

Scored against the tokens that actually follow, split into the immediate token
(offset 0) and the genuine future (offsets 1-15). A vector that carries workspace
content should beat the raw activation on the FUTURE tokens even if its cosine to
the local transport is low. If it does not, the averaged deep operator carries no
future content and the negative result is complete.
"""
import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="/workspace/data/spans_jvp/shard_3_jvp.parquet")
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--n-rows", type=int, default=512)
ap.add_argument("--topk", type=int, default=50)
ap.add_argument("--show", type=int, default=6)
ap.add_argument("--out", default="/workspace/results/multislot_eval/deep_logitlens.json")
args = ap.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
model = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)
inner = model.model if hasattr(model, "model") else model
NORM, WU = inner.norm, model.lm_head.weight

Jb = [torch.from_numpy(np.load(f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy")).float().to(dev)
      for d in range(16)]
nrm = [float(J.norm()) for J in Jb]
JD = sum((1.0 / nrm[d]) * Jb[d] for d in range(1, 16))     # norm-equalised deep pool
print(f"deep pool over 1<=d<16, norm-equalised (|JD|={float(JD.norm()):.2f})", flush=True)


def lens(v, k):
    h = NORM(v.to(WU.dtype).unsqueeze(0))
    return (h @ WU.T).float()[0].topk(k).indices.tolist()


t = pq.read_table(sorted(glob.glob(args.shards))[0],
                  columns=["activation_vector", "rollout_token_ids", "ctx_text",
                           "h42_recompute_cosine"])
rows = [r for r in t.slice(0, args.n_rows * 3).to_pylist()
        if r["h42_recompute_cosine"] >= 0.99 and len(r["rollout_token_ids"]) >= 16
        ][: args.n_rows]
print(f"{len(rows)} rows", flush=True)

HBAR = torch.from_numpy(
    np.load(f"{args.jbar_dir}/hbar_L42.npy")).float().to(dev)

# The deep pool shows a position-invariant basin (British-English / EU vocabulary
# recurring across unrelated contexts) on top of its real future content — the
# same disease as the "always talking about aircraft" readouts. Centering the
# INPUT is the principled fix and is exactly equivalent to removing the corpus
# mean from the output, since J(h - hbar) = Jh - J·hbar. Unlike the earlier
# centering attempt this changes only the test-time vector being inspected, so
# there is no train/test transformation mismatch to confound it.
CANDS = ("raw_h42", "jbar0", "deep_pooled", "jbar0_centered", "deep_pooled_centered")
hit = {c: {"cur": [], "fut": [], "shuf": []} for c in CANDS}
shown = []
for i, r in enumerate(rows):
    h = torch.tensor(np.array(r["activation_vector"], dtype=np.float32), device=dev)
    fut = [int(x) for x in r["rollout_token_ids"][:16]]
    cur_set, fut_set = {fut[0]}, set(fut[1:])
    hc = h - HBAR
    vecs = {"raw_h42": h, "jbar0": Jb[0] @ h, "deep_pooled": JD @ h,
            "jbar0_centered": Jb[0] @ hc, "deep_pooled_centered": JD @ hc}
    tops = {}
    for c in CANDS:
        top = lens(vecs[c], args.topk)
        tops[c] = top
        s = set(top)
        hit[c]["cur"].append(len(s & cur_set) / max(1, len(cur_set)))
        hit[c]["fut"].append(len(s & fut_set) / max(1, len(fut_set)))
        other = rows[(i + len(rows) // 2) % len(rows)]
        osl = set(int(x) for x in other["rollout_token_ids"][1:16])
        hit[c]["shuf"].append(len(s & osl) / max(1, len(osl)))
    if i < args.show:
        shown.append({
            "context_tail": r["ctx_text"][-140:],
            "actual_future": tok.decode(fut),
            "top": {c: [tok.decode([x]) for x in tops[c][:14]] for c in CANDS}})

res = {"n_rows": len(rows), "topk": args.topk,
       "recall": {c: {"immediate_token": float(np.mean(hit[c]["cur"])),
                      "future_tokens_1_15": float(np.mean(hit[c]["fut"])),
                      "shuffled_future": float(np.mean(hit[c]["shuf"])),
                      "future_above_shuffled": float(np.mean(hit[c]["fut"])
                                                     - np.mean(hit[c]["shuf"]))}
                  for c in CANDS},
       "examples": shown}

print(f"\nrecall of the tokens that actually follow, in the top-{args.topk}")
print(f"{'vector':22s} immediate  future(1-15)  SHUFFLED  future-shuffled")
for c in CANDS:
    v = res["recall"][c]
    print(f"{c:22s}   {v['immediate_token']:.3f}      {v['future_tokens_1_15']:.3f}"
          f"        {v['shuffled_future']:.3f}      {v['future_above_shuffled']:+.3f}")
print("\nThe last column is the only honest one: recall against a DIFFERENT row's"
      "\nfuture measures generic token frequency, and raw h42 decodes to frequent"
      "\njunk, which inflates its raw future recall.")

base = res["recall"]["raw_h42"]["future_above_shuffled"]
deep = max(res["recall"]["deep_pooled"]["future_above_shuffled"],
           res["recall"]["deep_pooled_centered"]["future_above_shuffled"])
res["deep_minus_raw_on_future"] = deep - base
print(f"\ndeep pooled beats raw h42 on FUTURE tokens by {deep - base:+.3f}")
print("VERDICT:", "the averaged deep operator DOES carry future content the "
      "activation lacks — worth a decoder" if deep - base > 0.01 else
      "no future content beyond the activation; the negative result is complete")

for s in shown[: args.show]:
    print(f"\n  ctx …{s['context_tail'][-90:]}")
    print(f"  actual next: {s['actual_future'][:80]!r}")
    for c in CANDS:
        print(f"    {c:12s} {' '.join(repr(x) for x in s['top'][c][:9])}")

json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=1)
print(f"\nwrote {args.out}", flush=True)
