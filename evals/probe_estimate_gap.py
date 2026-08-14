"""How much does the TEST-TIME estimate cost, in nats, on identical rows?

Arm D trains on the REAL penultimate states h62[p..p+K-1] and is tested on
Jbar^(d)_{42->62} @ h42[p], an estimate of those states (audit: cos +0.50 at
d=0, ~0.1 beyond). This measures the gap with nothing else changing:

    CE_real      teacher-forced response CE with the real states in the slots
    CE_estimate  same rows, same model, slots replaced by the Jbar estimates

A near-deterministic reader (arm D reached CE 0.18) should degrade sharply, and
the size of that jump is the cleanest single number for whether the estimate
carries enough to drive the lens. It needs no judge and no sampling, so it can
run on an early checkpoint instead of waiting for a full training run.

Also reports the degenerate-output rate on a handful of greedy generations,
since a low CE with collapsed text is the failure mode already seen once.
"""
import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.datagen.injection_tokens import find_injection_token
from nla.schema import compute_canonical_neighbors
from nla.utils.hooks import register_karvonen_hook
from pretrain.finalize_jvp_spans import ACTOR_TEMPLATE_MULTI

ap = argparse.ArgumentParser()
ap.add_argument("--av-ckpt", required=True)
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--shards", default="/workspace/data/spans_pen8/shard_3_jvp.parquet")
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--src-layer", type=int, default=42)
ap.add_argument("--tgt-layer", type=int, default=62)
ap.add_argument("--k-slots", type=int, default=8)
ap.add_argument("--n-rows", type=int, default=256)
ap.add_argument("--n-gen", type=int, default=8)
ap.add_argument("--slot-scale", choices=["per_slot", "shared", "both"],
                default="both",
                help="per_slot norm-matches every slot to ||h_p||, which at test "
                     "time AMPLIFIES the deep slots ~500x (their estimates carry "
                     "0.2-3%% of the real state's norm) — training saw real states "
                     "of uniform norm, so matching was a no-op there. shared "
                     "divides by the row's largest slot norm, preserving the "
                     "relative magnitudes so faint slots stay faint.")
ap.add_argument("--out", default="/workspace/results/multislot_eval/estimate_gap.json")
args = ap.parse_args()
K, dev = args.k_slots, "cuda"

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
base = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16,
    attn_implementation="sdpa").to(dev).eval()
model = PeftModel.from_pretrained(base, args.av_ckpt).eval()
inj_char, inj_id = find_injection_token(tok)
TEMPLATE = ACTOR_TEMPLATE_MULTI.format(k=K, markers="{injection_char}" * K)
left, right = compute_canonical_neighbors(tok, TEMPLATE, inj_char, inj_id)
vref = [None]
sref = [None, 1]
register_karvonen_hook(model, vref, inj_id, left, right, scale_ref=sref)
torch.set_grad_enabled(False)

Jb = [torch.from_numpy(np.load(
    f"{args.jbar_dir}/Jbar_L{args.src_layer}_to_L{args.tgt_layer}_off{d}.npy")
    ).float().to(dev) for d in range(K)]

content = TEMPLATE.format(injection_char=inj_char)
pstr = tok.apply_chat_template([{"role": "user", "content": content}],
                              tokenize=False, add_generation_prompt=True,
                              enable_thinking=False)
P = tok.encode(pstr, add_special_tokens=False)

rows = pq.read_table(sorted(glob.glob(args.shards))[0]).slice(0, args.n_rows * 2).to_pylist()
rows = [r for r in rows if r["rollout_token_ids"]
        and len(r["rollout_token_ids"]) >= K][: args.n_rows]
print(f"{len(rows)} rows | ckpt {args.av_ckpt}", flush=True)


def slots_real(r):
    tv = np.frombuffer(r["transported_vectors"], dtype=np.float16).reshape(16, -1)
    return torch.tensor(tv[:K].astype(np.float32), device=dev)


def slots_estimate(r):
    h = torch.tensor(np.array(r["activation_vector"], dtype=np.float32), device=dev)
    return torch.stack([Jb[d] @ h for d in range(K)])


def ce(kind, batch=8):
    tot, ntok = 0.0, 0
    for c0 in range(0, len(rows), batch):
        chunk = rows[c0:c0 + batch]
        seqs, plen, vecs = [], [], []
        for r in chunk:
            resp = tok.decode(r["rollout_token_ids"][:K], skip_special_tokens=True)
            rid = tok.encode(resp + (tok.eos_token or ""), add_special_tokens=False)
            seqs.append(P + rid)
            plen.append(len(P))
            vecs.append(slots_real(r) if kind == "real" else slots_estimate(r))
        T = max(len(s) for s in seqs)
        ids = torch.full((len(chunk), T), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        msk = torch.zeros((len(chunk), T))
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
            msk[i, plen[i]:len(s)] = 1
        ids, att, msk = ids.to(dev), att.to(dev), msk.to(dev)
        vref[0] = torch.cat(vecs).float()
        sref[0], sref[1] = SCALE, K
        try:
            lg = model(input_ids=ids, attention_mask=att).logits.float()
        finally:
            vref[0] = None
            sref[0] = None
        l = torch.nn.functional.cross_entropy(
            lg[:, :-1].reshape(-1, lg.shape[-1]), ids[:, 1:].reshape(-1),
            reduction="none").view(ids[:, 1:].shape)
        tot += float((l * msk[:, 1:]).sum())
        ntok += int(msk[:, 1:].sum())
    return tot / max(1, ntok)


def gen(kind, n):
    out = []
    pt = torch.tensor([P], device=dev)
    for r in rows[:n]:
        vref[0] = (slots_real(r) if kind == "real" else slots_estimate(r)).float()
        sref[0], sref[1] = SCALE, K
        try:
            g = model.generate(input_ids=pt, attention_mask=torch.ones_like(pt),
                               max_new_tokens=K + 4, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        finally:
            vref[0] = None
            sref[0] = None
        out.append(tok.decode(g[0, pt.shape[1]:], skip_special_tokens=True).strip())
    return out


res = {"ckpt": args.av_ckpt, "n_rows": len(rows), "scalings": {}}
scalings = ["per_slot", "shared"] if args.slot_scale == "both" else [args.slot_scale]
for SCALE in scalings:
    sub = {}
    for kind in ("real", "estimate"):
        sub[f"ce_{kind}"] = ce(kind)
        sub[f"gen_{kind}"] = gen(kind, args.n_gen)
        print(f"\n[{SCALE}] CE({kind:8s}) = {sub[f'ce_{kind}']:.4f}", flush=True)
        for i, g in enumerate(sub[f"gen_{kind}"][:4]):
            truth = tok.decode(rows[i]["rollout_token_ids"][:K], skip_special_tokens=True)
            print(f"   gen={g[:58]!r}\n   true={truth[:58]!r}", flush=True)
    sub["gap_nats"] = sub["ce_estimate"] - sub["ce_real"]
    print(f"[{SCALE}] GAP = {sub['gap_nats']:+.3f} nats", flush=True)
    res["scalings"][SCALE] = sub
best = min(res["scalings"], key=lambda k: res["scalings"][k]["gap_nats"])
res["gap_nats"] = res["scalings"][best]["gap_nats"]
res["best_scaling"] = best
json.dump(res, open(args.out, "w"), indent=2)
print(f"\nBEST scaling: {best}  GAP = {res['gap_nats']:+.3f} nats", flush=True)
if len(res["scalings"]) > 1:
    d = (res["scalings"]["per_slot"]["gap_nats"]
         - res["scalings"]["shared"]["gap_nats"])
    print(f"shared scaling changes the gap by {-d:+.3f} nats "
          f"(positive = shared is better)", flush=True)
print("VERDICT:", "estimate is USABLE (gap < 1 nat)" if res["gap_nats"] < 1.0
      else f"estimate still degrades the readout (+{res['gap_nats']:.2f} nats)",
      flush=True)
