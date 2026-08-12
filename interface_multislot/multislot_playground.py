"""Multi-slot workspace-lens playground (STANDALONE COPY — does not touch
interface/, which belongs to a sibling session).

Adapted from interface/weirdchat_lens.py for the K=8 per-offset transported
lens (arm A) + its single-slot baseline (arm C):

  * paste/run a prompt (system+user+assistant supported), click any token
  * linear readouts: logit lens / pooled-J̄ lens at L42 and L62
  * trained readouts via the comparison slots: category "armA-K8" exposes every
    slot construction from evals/slot_builders.py as the checkpoint dropdown,
    ordered best-measured-first (judged workspace agreement, 551 items, 0-2):
      diff 0.586                slot 0 intact, slots 1-7 = horizon increments
      keep0_deflate_rest        slot 0 intact, 1-7 shared-component removed
      keep0_gs_rest             slot 0 intact, 1-7 orthogonalized
      per_offset 0.289          the design as specified: slot d = Jbar^(d) @ h42
      slot0_only 0.544 / shuffled 0.532 / no_slot0 0.497   knockouts
      pooled_identical 0.269 / deflated 0.236 / centered 0.218
    category "armC-1slot" runs the single-slot baseline on Jbar_pooled @ h42.

  PORT=8807 CUDA_VISIBLE_DEVICES=1 python multislot_playground.py
"""
import glob, json, os, re, secrets, sys, threading, time

import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "interface"))
sys.path.insert(0, os.path.join(_HERE, "..", "evals"))
sys.path.insert(0, "/workspace/skip-lens/interface")
sys.path.insert(0, "/workspace/skip-lens/evals")
from common import resolve_text_model, norm_gain
from slot_builders import CONDITIONS, build_slots

_USER = os.environ.get("PG_USER", "claude")
_PASS = os.environ.get("PG_PASS", "claube")
_sec = HTTPBasic()


def require_auth(c: HTTPBasicCredentials = Depends(_sec)):
    if not (secrets.compare_digest(c.username, _USER)
            and secrets.compare_digest(c.password, _PASS)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad creds",
                            headers={"WWW-Authenticate": "Basic"})


BASE = os.environ.get("BASE_CKPT", "Qwen/Qwen3.6-27B")
JBAR_DIR = os.environ.get("JBAR_DIR", "/workspace/results/offset_jlens")
ARMA_CKPT = os.environ.get("ARMA_CKPT",
    "/workspace/skip-lens/ckpts/multislot_armA_k8/iter_0003875")
ARMC_CKPT = os.environ.get("ARMC_CKPT",
    "/workspace/skip-lens/ckpts/multislot_armC_L62/iter_0003876")
TARGET_L = 62
SRC_L = 42
K = 8
PORT = int(os.environ.get("PORT", 8807))
dev = "cuda"

print("[ms] loading model ...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
tm = resolve_text_model(model)
W_U = model.lm_head.weight.detach()
GAIN = norm_gain(model).to(dev)
EPS = float(getattr(tm.norm, "variance_epsilon", getattr(tm.norm, "eps", 1e-6)))

JBAR = [torch.from_numpy(np.load(
    os.path.join(JBAR_DIR, f"Jbar_L{SRC_L}_to_L{TARGET_L}_off{d}.npy"))
    ).to(dev).float() for d in range(K)]
JPOOL = torch.from_numpy(np.load(
    os.path.join(JBAR_DIR, f"Jbar_L{SRC_L}_to_L{TARGET_L}_offpooled.npy"))).to(dev).float()
JMATS = {SRC_L: JPOOL.to(torch.bfloat16)}
_hb = os.path.join(JBAR_DIR, f"hbar_L{SRC_L}.npy")
HBAR = (torch.from_numpy(np.load(_hb)).float().to(dev) if os.path.exists(_hb) else None)
MEANDIR = {}
for _d in range(K):
    _p = os.path.join(JBAR_DIR, f"meandir_off{_d}.npy")
    if os.path.exists(_p):
        MEANDIR[_d] = torch.from_numpy(np.load(_p)).float().to(dev)
print(f"[ms] centering assets: hbar={'yes' if HBAR is not None else 'NO'} "
      f"meandirs={len(MEANDIR)}", flush=True)
JLAYERS = [SRC_L]
CAP_LAYERS = [SRC_L, TARGET_L]
print(f"[ms] {K} per-offset Jbar + pooled loaded", flush=True)

from peft import PeftModel
from nla.datagen.injection_tokens import find_injection_token
from nla.schema import compute_canonical_neighbors
from nla.utils.hooks import register_karvonen_hook
from pretrain.finalize_jvp_spans import ACTOR_TEMPLATE_MULTI

SINGLE_TEMPLATE = ("You are shown an internal activation vector captured from a language model "
    "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
    "taken at one position and encodes what the model is about to generate next. "
    "Output the text the model most likely produces immediately after this point.\n\n"
    "<concept>{injection_char}</concept>")
MULTI_TEMPLATE = ACTOR_TEMPLATE_MULTI.format(k=K, markers="{injection_char}" * K)

PEFT = PeftModel.from_pretrained(model, ARMA_CKPT, adapter_name="armA").eval()
PEFT.load_adapter(ARMC_CKPT, adapter_name="armC")
inj_char, inj_id = find_injection_token(tok)
# canonical flanks are identical for a run of 1 and a run of K (same tag tokens)
_left, _right = compute_canonical_neighbors(tok, MULTI_TEMPLATE, inj_char, inj_id)
_vref = [None]
register_karvonen_hook(PEFT, _vref, inj_id, _left, _right)


def _pt(template):
    s = tok.apply_chat_template(
        [{"role": "user", "content": template.format(injection_char=inj_char)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return torch.tensor([tok.encode(s, add_special_tokens=False)], device=dev)


_PT_MULTI = _pt(MULTI_TEMPLATE)
_PT_SINGLE = _pt(SINGLE_TEMPLATE)
print("[ms] armA + armC adapters loaded", flush=True)

# Ordered so the first entries are the best-scoring constructions (judged
# workspace agreement, 551 official items, judge scale 0-2):
#   diff 0.586 | slot0_only 0.544 | shuffled 0.532 | no_slot0 0.497
#   per_offset 0.289 (the design as specified) | pooled_identical 0.269
#   deflated 0.236 | centered 0.218      keep0_* are the newest, untested here
COND_ORDER = ["diff", "keep0_deflate_rest", "keep0_gs_rest", "per_offset",
              "slot0_only", "shuffled_slots", "no_slot0", "gs",
              "pooled_identical", "deflated", "centered"]
COND_ORDER = [c for c in COND_ORDER if c in CONDITIONS]
REGISTRY = {"armA-K8": {c: c for c in COND_ORDER},
            "armC-1slot": {"pooledJ": "pooledJ"}}

RUNS, LOCK = {}, threading.Lock()


def slots_for(cond, h42):
    """Delegates to evals/slot_builders.py — the same code path the judged
    evals use, so the playground can never drift from the measured numbers."""
    return build_slots(cond, h42, JBAR, JPOOL, k=K, hbar=HBAR, meandirs=MEANDIR)


@torch.no_grad()
def _roll(vectors, adapter, prompt_ids, n=1, max_new=14, temp=0.7):
    """vectors: [S, d] (S = K for armA, 1 for armC), tiled per batch row."""
    B = max(1, n)
    ids = prompt_ids.repeat(B, 1)
    PEFT.set_adapter(adapter)
    _vref[0] = vectors.float().repeat(B, 1).contiguous()
    try:
        g = PEFT.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                          min_new_tokens=max_new, max_new_tokens=max_new,
                          do_sample=(temp > 0), temperature=max(temp, 1e-5),
                          top_p=0.95, pad_token_id=tok.eos_token_id)
    finally:
        _vref[0] = None
    return [tok.decode(x[prompt_ids.shape[1]:], skip_special_tokens=True).strip()
            for x in g]


def unit_rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


@torch.no_grad()
def readout(h, head, use_j, layer, topk=10, feed=""):
    z = h.to(torch.bfloat16)
    mode = feed or ("j" if use_j else "raw")
    if mode == "j" and layer in JMATS:
        z = z @ JMATS[layer].T
    hidden = z.float()
    logits = ((unit_rms(hidden) * GAIN).to(torch.bfloat16) @ W_U.T).float()
    p = torch.softmax(logits, -1)
    v, i = p.topk(topk, -1)
    return i.cpu().tolist(), v.cpu().tolist()


class RunReq(BaseModel):
    user: str = ""
    assistant: str = ""
    system: str = ""
    text: str = ""
    chat: bool = True
    max_new_tokens: int = 0
    temperature: float = 1.0


class ReadReq(BaseModel):
    run_id: str
    layer: int
    use_j: bool = True
    head: str = "lm"
    topk: int = 10
    feed: str = ""


class LensSlot(BaseModel):
    cat: str
    ckpt: str
    max_new: int = 0


class AoReq(BaseModel):
    run_id: str
    pos: int
    layer: int
    use_j: bool = True
    n: int = 1
    max_new: int = 20
    cnla_ckpt: str = ""
    lenses: list[str] | None = None
    feed: str = ""
    slots: list[LensSlot] | None = None


app = FastAPI()


@app.get("/api/presets")
def presets(_=Depends(require_auth)):
    return JSONResponse({"behaviors": {}, "layers": JLAYERS,
                         "target": TARGET_L, "heads": []})


@app.get("/api/registry")
def api_registry(_=Depends(require_auth)):
    return {"categories": {cat: list(ckpts.keys()) for cat, ckpts in REGISTRY.items()}}


@app.post("/api/run")
@torch.no_grad()
def run(r: RunReq, _=Depends(require_auth)):
    def _ids(x):
        return x if isinstance(x, torch.Tensor) else x["input_ids"]

    sysmsg = [{"role": "system", "content": r.system}] if r.system else []
    if r.assistant:
        pre = _ids(tok.apply_chat_template(sysmsg + [{"role": "user", "content": r.user}],
                                           add_generation_prompt=True, return_tensors="pt"))
        rep = tok(r.assistant, add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([pre, rep], 1).to(dev)
        n_prompt = pre.shape[1]
        gen = ids
    else:
        if r.chat:
            ids = _ids(tok.apply_chat_template(sysmsg + [{"role": "user", "content": r.text or r.user}],
                                               add_generation_prompt=True, return_tensors="pt"))
        else:
            ids = tok(r.text or r.user, return_tensors="pt").input_ids
        ids = ids.to(dev)
        n_prompt = ids.shape[1]
        n_new = max(0, int(r.max_new_tokens))
        gen = ids if n_new == 0 else model.generate(
            ids, attention_mask=torch.ones_like(ids), do_sample=r.temperature > 0,
            temperature=max(r.temperature, 1e-5), top_k=0, top_p=1.0,
            max_new_tokens=n_new, pad_token_id=tok.eos_token_id or 0)

    grab = {}
    hs = [tm.layers[L].register_forward_hook(
        (lambda L: (lambda m, i, o: grab.__setitem__(
            L, (o[0] if isinstance(o, tuple) else o)[0].detach())))(L))
        for L in CAP_LAYERS]
    with PEFT.disable_adapter():
        model(gen)
    for h in hs:
        h.remove()

    toks = [tok.decode([int(t)]) for t in gen[0]]
    rid = secrets.token_hex(8)
    with LOCK:
        RUNS[rid] = {"acts": {L: grab[L].clone() for L in CAP_LAYERS},
                     "tokens": toks, "n_prompt": n_prompt, "ts": time.time()}
        for k in [k for k, v in RUNS.items() if time.time() - v["ts"] > 3600]:
            RUNS.pop(k, None)
    return {"run_id": rid, "tokens": toks, "n_prompt": n_prompt,
            "n_total": len(toks)}


@app.post("/api/readout")
def read(r: ReadReq, _=Depends(require_auth)):
    with LOCK:
        run = RUNS.get(r.run_id)
    if not run:
        raise HTTPException(404, "run expired — re-run the prompt")
    layer = r.layer if r.layer in CAP_LAYERS else SRC_L
    ids, ps = readout(run["acts"][layer], r.head, r.use_j, layer, r.topk, feed=r.feed)
    return {"tokens": run["tokens"], "n_prompt": run["n_prompt"],
            "top": [[[tok.decode([t]), round(p, 4)] for t, p in zip(a, b)]
                    for a, b in zip(ids, ps)]}


class AllReq(BaseModel):
    run_id: str
    pos: int
    layer: int
    use_j: bool = True
    topk: int = 10
    feed: str = ""


@app.post("/api/allheads")
def allheads(r: AllReq, _=Depends(require_auth)):
    """No fitted horizon heads here — one row per linear readout variant
    (pooled-J̄ transported vs raw) so the UI's head matrix still renders."""
    with LOCK:
        run = RUNS.get(r.run_id)
    if not run:
        raise HTTPException(404, "run expired -- reload the transcript")
    layer = r.layer if r.layer in CAP_LAYERS else SRC_L
    h = run["acts"][layer][r.pos:r.pos + 1]
    k = max(1, min(int(r.topk), 50))
    rows = []
    for label, feed in [("lm", r.feed or ("j" if r.use_j else "raw"))]:
        i, p = readout(h, "lm", r.use_j, layer, k, feed=feed)
        rows.append({"head": label,
                     "top": [[tok.decode([t]), round(v, 4)] for t, v in zip(i[0], p[0])]})
    toks = run["tokens"]
    return {"pos": r.pos, "rows": rows,
            "actual_next": toks[r.pos + 1:r.pos + 9],
            "token": toks[r.pos],
            "is_assistant": r.pos >= run["n_prompt"]}


@app.get("/api/cnla_ckpts")
def api_cnla_ckpts(_=Depends(require_auth)):
    return {"ckpts": [], "default": "n/a (multi-slot playground)"}


@app.post("/api/ao")
def api_ao(r: AoReq, _=Depends(require_auth)):
    with LOCK:
        run = RUNS.get(r.run_id)
    if not run:
        raise HTTPException(404, "run expired -- reload the transcript")
    h42 = run["acts"][SRC_L][r.pos].float()
    n, mx = max(1, min(int(r.n), 6)), max(4, min(int(r.max_new), 48))
    slots_req = r.slots if r.slots is not None else [
        LensSlot(cat="armA-K8", ckpt="per_offset"),
        LensSlot(cat="armA-K8", ckpt="pooled_identical"),
        LensSlot(cat="armC-1slot", ckpt="pooledJ"),
    ]
    out_slots = []
    with LOCK:
        for s in slots_req[:3]:
            cell = {"cat": s.cat, "ckpt": s.ckpt}
            try:
                smx = max(1, min(int(s.max_new), 64)) if s.max_new else mx
                if s.cat == "armA-K8":
                    if s.ckpt not in COND_ORDER:
                        raise HTTPException(404, f"unknown condition {s.ckpt!r}")
                    cell["readout"] = _roll(slots_for(s.ckpt, h42), "armA",
                                            _PT_MULTI, n, smx)
                elif s.cat == "armC-1slot":
                    cell["readout"] = _roll((JPOOL @ h42).unsqueeze(0), "armC",
                                            _PT_SINGLE, n, smx)
                else:
                    raise HTTPException(404, f"unknown category {s.cat!r}")
                cell["max_new"] = smx
            except HTTPException as e:
                cell["readout"], cell["error"] = [], str(e.detail)
            except Exception as e:
                cell["readout"], cell["error"] = [], repr(e)[:200]
            out_slots.append(cell)
    return {"pos": r.pos, "layer": SRC_L, "feed": "jbar", "slots": out_slots}


@app.get("/", response_class=HTMLResponse)
def index(_=Depends(require_auth)):
    return open(UI_FILE).read()


UI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "multislot_ui.html")

if __name__ == "__main__":
    _z = torch.zeros(W_U.shape[1], device=dev)
    for _ad, _pt_, _S in [("armA", _PT_MULTI, K), ("armC", _PT_SINGLE, 1)]:
        try:
            _roll(torch.zeros(_S, W_U.shape[1], device=dev), _ad, _pt_, n=1, max_new=14)
            print(f"[ms] warmup: {_ad} compiled", flush=True)
        except Exception as e:
            print(f"[ms] warmup {_ad} skipped: {e!r}", flush=True)
    print(f"[ms] serving on 0.0.0.0:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
