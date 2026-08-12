"""Jacobian-leakage probe: arm A heldout-style CE on WRONG-TANGENT transports.

Probe rows (collect_jvp_transport --probe-frac) carry transports
J_local(this context) @ h_other — the local Jacobian of THIS rollout applied
to a DIFFERENT row's activation. If the multi-slot AV can still predict this
row's rollout from them, the training signal leaked through the context-
dependent Jacobian rather than the activation. Healthy result: probe CE
near the unconditional-prior CE, far above real-row CE.

Usage:
  python evals/leakage_probe_ce.py --av-ckpt ckpts/multislot_armA_k8/iter_0003875 \
      --probe-glob '/workspace/data/spans_jvp/shard_*_jvp_probe.parquet' \
      --real-glob  '/workspace/data/spans_jvp/shard_0_jvp.parquet' --n-rows 512
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

K = 8

ap = argparse.ArgumentParser()
ap.add_argument("--av-ckpt", required=True)
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--probe-glob", required=True)
ap.add_argument("--real-glob", required=True)
ap.add_argument("--n-rows", type=int, default=512)
ap.add_argument("--out", default="/workspace/results/multislot_eval/leakage_probe.json")
args = ap.parse_args()
dev = "cuda"

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
base = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16,
    attn_implementation="sdpa").to(dev).eval()
model = PeftModel.from_pretrained(base, args.av_ckpt).eval()
inj_char, inj_id = find_injection_token(tok)
template = ACTOR_TEMPLATE_MULTI.format(k=K, markers="{injection_char}" * K)
left, right = compute_canonical_neighbors(tok, template, inj_char, inj_id)
vref = [None]
register_karvonen_hook(model, vref, inj_id, left, right)
torch.set_grad_enabled(False)

content = template.format(injection_char=inj_char)
prompt_str = tok.apply_chat_template(
    [{"role": "user", "content": content}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False)
prompt_ids = tok.encode(prompt_str, add_special_tokens=False)


def rows_from(glob_pat, n):
    out = []
    for f in sorted(glob.glob(glob_pat)):
        t = pq.read_table(f, columns=["rollout_token_ids", "transported_vectors"])
        out.extend(t.to_pylist())
        if len(out) >= n:
            break
    return out[:n]


def mean_ce(rows, batch=16):
    tot, ntok = 0.0, 0
    for c0 in range(0, len(rows), batch):
        chunk = rows[c0:c0 + batch]
        seqs, plens, vecs = [], [], []
        for r in chunk:
            resp = tok.decode(r["rollout_token_ids"][:K], skip_special_tokens=True)
            resp_ids = tok.encode(resp + (tok.eos_token or ""), add_special_tokens=False)
            seqs.append(prompt_ids + resp_ids)
            plens.append(len(prompt_ids))
            tv = np.frombuffer(r["transported_vectors"], dtype=np.float16)
            vecs.append(tv.reshape(16, -1)[:K].astype(np.float32))
        T = max(len(s) for s in seqs)
        ids = torch.full((len(chunk), T), tok.eos_token_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        msk = torch.zeros((len(chunk), T))
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
            msk[i, plens[i]:len(s)] = 1
        ids, att, msk = ids.to(dev), att.to(dev), msk.to(dev)
        vref[0] = torch.tensor(np.concatenate(vecs), dtype=torch.float32, device=dev)
        try:
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        finally:
            vref[0] = None
        sl = logits[:, :-1]
        st = ids[:, 1:]
        sm = msk[:, 1:]
        ce = torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.shape[-1]), st.reshape(-1), reduction="none"
        ).view(st.shape)
        tot += float((ce * sm).sum())
        ntok += int(sm.sum())
    return tot / max(1, ntok)


real = rows_from(args.real_glob, args.n_rows)
probe = rows_from(args.probe_glob, args.n_rows)
ce_real = mean_ce(real)
ce_probe = mean_ce(probe)
res = {"ce_real": ce_real, "ce_probe": ce_probe,
       "gap_nats": ce_probe - ce_real,
       "n_real": len(real), "n_probe": len(probe)}
json.dump(res, open(args.out, "w"), indent=2)
print(json.dumps(res, indent=2), flush=True)
print("VERDICT:", "NO MEANINGFUL LEAK (probe >> real)" if ce_probe - ce_real > 1.0
      else "POSSIBLE LEAK — probe CE close to real, investigate", flush=True)
