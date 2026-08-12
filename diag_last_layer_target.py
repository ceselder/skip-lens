"""Go/no-go: does targeting the LAST block beat the penultimate one?

The shipped per-offset family maps L42 -> L62 (penultimate, the paper's
convention). This measures the same family fitted to L42 -> L63 (last block),
where the residual is maximally output-aligned (W_U . norm(h_last) IS the
model's prediction, with no approximation).

Metric: top-1 / top-5 agreement of the lens readout with the model's OWN next
token, on held-out text, against three references:
  plain logit lens at L42          (audit measured 10.3% top-1)
  penultimate-target Jbar^(0)      (audit measured  6.8% top-1)
  oracle: logit lens on true h_tgt (upper bound)
Also reports the per-horizon version: does Jbar^(d) predict the token at
t+d+1 better under the last-block target?

Uses the model's own (1+weight) RMSNorm gain via interface/common.norm_gain.
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/skip-lens/interface")
from common import norm_gain, resolve_text_model  # noqa: E402

BASE = "Qwen/Qwen3.6-27B"
SRC = 42
PEN_DIR = os.environ.get("PEN_DIR", "/workspace/results/offset_jlens")
LAST_DIR = os.environ.get("LAST_DIR", "/workspace/results/offset_jlens_last")
PEN_L = int(os.environ.get("PEN_L", "62"))
LAST_L = int(os.environ.get("LAST_L", "63"))
N_PROMPTS = int(os.environ.get("N_PROMPTS", "16"))
SEQ = int(os.environ.get("SEQ", "512"))
DELTAS = [int(x) for x in os.environ.get("DELTAS", "0,1,2,3").split(",")]
OUT = os.environ.get("OUT", "/workspace/results/multislot_eval/last_layer_target.json")

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
tm = resolve_text_model(model)
W_U = model.lm_head.weight.detach()
GAIN = norm_gain(model).cuda()
EPS = float(getattr(tm.norm, "variance_epsilon", getattr(tm.norm, "eps", 1e-6)))
torch.set_grad_enabled(False)


def unit_rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def readout_ids(vecs, k=5):
    """vecs [N, d] -> top-k token ids under the model's own head."""
    logits = ((unit_rms(vecs.float()) * GAIN).to(torch.bfloat16) @ W_U.T).float()
    return logits.topk(k, -1).indices


def load_family(d, tgt):
    out = {}
    for dl in DELTAS:
        p = f"{d}/Jbar_L{SRC}_to_L{tgt}_off{dl}.npy"
        if os.path.exists(p):
            out[dl] = torch.from_numpy(np.load(p)).float().cuda()
    return out


PEN = load_family(PEN_DIR, PEN_L)
LAST = load_family(LAST_DIR, LAST_L)
print(f"penultimate-target offsets: {sorted(PEN)} | last-target offsets: {sorted(LAST)}",
      flush=True)
if not LAST:
    raise SystemExit(f"no last-layer matrices in {LAST_DIR} — fit them first")

pool = pq.read_table("/workspace/data/ffw_fit_pool.parquet",
                     columns=["text"]).column("text").to_pylist()
grab = {}
for L in (SRC, PEN_L, LAST_L):
    tm.layers[L].register_forward_hook(
        (lambda L: (lambda m, i, o: grab.__setitem__(
            L, (o[0] if isinstance(o, tuple) else o)[0].detach())))(L))

stats = {k: {"t1": 0, "t5": 0, "n": 0} for k in
         ("logit_L42", "pen_off0", "last_off0", "oracle_pen", "oracle_last")}
per_h = {dl: {"pen": {"t1": 0, "n": 0}, "last": {"t1": 0, "n": 0}} for dl in DELTAS}

used = 0
for text in pool:
    if used >= N_PROMPTS:
        break
    ids = tok(text, return_tensors="pt", truncation=True, max_length=SEQ).input_ids.cuda()
    if ids.shape[1] < 64:
        continue
    used += 1
    model(input_ids=ids)
    h42, hp, hl = grab[SRC], grab[PEN_L], grab[LAST_L]
    T = ids.shape[1]
    pos = torch.arange(24, T - max(DELTAS) - 2, device=ids.device)   # skip sinks
    gold_next = ids[0, pos + 1]

    for name, vecs in (("logit_L42", h42[pos]),
                       ("pen_off0", (PEN[0] @ h42[pos].T).T if 0 in PEN else None),
                       ("last_off0", (LAST[0] @ h42[pos].T).T if 0 in LAST else None),
                       ("oracle_pen", hp[pos]), ("oracle_last", hl[pos])):
        if vecs is None:
            continue
        top = readout_ids(vecs)
        stats[name]["t1"] += int((top[:, 0] == gold_next).sum())
        stats[name]["t5"] += int((top == gold_next[:, None]).any(-1).sum())
        stats[name]["n"] += len(pos)

    for dl in DELTAS:
        gold_d = ids[0, pos + dl + 1]
        for tag, fam in (("pen", PEN), ("last", LAST)):
            if dl not in fam:
                continue
            top = readout_ids((fam[dl] @ h42[pos].T).T, k=1)
            per_h[dl][tag]["t1"] += int((top[:, 0] == gold_d).sum())
            per_h[dl][tag]["n"] += len(pos)
    if used % 4 == 0:
        print(f"  {used}/{N_PROMPTS} prompts", flush=True)

res = {"n_prompts": used, "next_token": {}, "per_horizon": {}}
print(f"\n=== next-token agreement ({used} prompts, {stats['logit_L42']['n']} positions)")
for k, v in stats.items():
    if v["n"]:
        res["next_token"][k] = {"top1": v["t1"] / v["n"], "top5": v["t5"] / v["n"]}
        print(f"  {k:12s} top1={100*v['t1']/v['n']:5.1f}%  top5={100*v['t5']/v['n']:5.1f}%",
              flush=True)
print("\n=== per-horizon: predicting the token at t+d+1")
for dl in DELTAS:
    row = {}
    for tag in ("pen", "last"):
        if per_h[dl][tag]["n"]:
            row[tag] = per_h[dl][tag]["t1"] / per_h[dl][tag]["n"]
    res["per_horizon"][dl] = row
    if row:
        print("  d=%d  " % dl + "  ".join(f"{t}={100*x:5.1f}%" for t, x in row.items()),
              flush=True)

json.dump(res, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}", flush=True)
nt = res["next_token"]
if "last_off0" in nt and "logit_L42" in nt:
    gain = nt["last_off0"]["top1"] - nt["logit_L42"]["top1"]
    print("VERDICT:", "LAST-LAYER TARGET WINS — beats the plain logit lens "
          f"(+{100*gain:.1f}pp top-1); worth the full arm" if gain > 0 else
          f"last-layer target still loses to the logit lens ({100*gain:+.1f}pp); "
          "averaged Jacobians are weak on this model regardless of target",
          flush=True)
