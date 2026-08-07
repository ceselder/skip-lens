"""WeirdChat multi-token lens playground.

Browse Transluce/WeirdChat prompts (or paste your own), run Qwen3.6-27B, then pick
INDEPENDENTLY:

  * the JACOBIAN   J_{L->62}  for any source layer L that has a matrix (or "none")
  * the LM HEAD    the model's own W_U, or a fitted horizon-k head A_k

and read the lens out at every position:

    lens(h_L) = softmax( W_U . norm( A_k . ( J_{L->62} . h_L ) ) )

Both A_k and every J target block 62, so the head x Jacobian cross-product is valid
for every source layer. Setting head = "LM head" and J = on reproduces the ordinary
J-lens; head = "LM head", J = off reproduces the plain logit lens.

The model runs ONCE per prompt; activations for all J-source layers are cached, so
switching head/Jacobian/layer afterwards is instant.

  BASE_CKPT=... JDIR=... HEADS=... PORT=8802 python scripts/weirdchat_lens.py
"""
import glob, json, os, re, secrets, sys, threading, time
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import resolve_text_model, norm_gain

_USER = os.environ.get("PG_USER", "claude")
_PASS = os.environ.get("PG_PASS", "claube")
_sec = HTTPBasic()


def require_auth(c: HTTPBasicCredentials = Depends(_sec)):
    if not (secrets.compare_digest(c.username, _USER)
            and secrets.compare_digest(c.password, _PASS)):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad creds",
                            headers={"WWW-Authenticate": "Basic"})


BASE = os.environ.get("BASE_CKPT", "Qwen/Qwen3.6-27B")
JDIR = os.environ.get("JDIR", "/workspace-vast/celeste/multi-token-jlens-nla-lastlayer/"
                              "results/jlens")
HEADS = os.environ.get("HEADS", "/workspace-vast/celeste/regression-futurelens/"
                                "results/mtj_probes")
PRESETS = os.environ.get("PRESETS", "/workspace-vast/celeste/"
                                    "multi-token-jlens-nla-lastlayer/data/weirdchat_qwen.json")
TARGET_L = int(os.environ.get("TARGET_L", 62))
PORT = int(os.environ.get("PORT", 8802))
dev = "cuda"

JLAYERS = sorted(int(re.search(rf"J_L(\d+)_to_L{TARGET_L}", os.path.basename(p)).group(1))
                 for p in glob.glob(os.path.join(JDIR, f"J_L*_to_L{TARGET_L}.npy")))
print(f"[wc] J source layers {JLAYERS} -> block {TARGET_L}", flush=True)

print("[wc] loading model ...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
tm = resolve_text_model(model)
W_U = model.lm_head.weight.detach()
GAIN = norm_gain(model).to(dev)
EPS = float(getattr(tm.norm, "variance_epsilon", getattr(tm.norm, "eps", 1e-6)))
print(f"[wc] model ready. d={W_U.shape[1]} V={W_U.shape[0]}", flush=True)

JMATS = {L: torch.from_numpy(np.load(os.path.join(JDIR, f"J_L{L}_to_L{TARGET_L}.npy"))
                             ).to(dev).to(torch.bfloat16) for L in JLAYERS}

HEADW = {}
for f in sorted(glob.glob(os.path.join(HEADS, "A62_k*.pt"))):
    k = int(re.search(r"A62_k(\d+)", f).group(1))
    w = torch.load(f, map_location=dev)
    HEADW[k] = (w["A"].to(dev).float(), w["b"].to(dev).float(),
                float(w["log_s"].exp()) if "log_s" in w else 1.0,
                w.get("xmode", "unit"))
print(f"[wc] horizon heads: {sorted(HEADW)}", flush=True)

PRESET_JSON = json.load(open(PRESETS)) if os.path.exists(PRESETS) else {}
CAP_LAYERS = sorted(set(JLAYERS) | {TARGET_L})
RUNS, LOCK = {}, threading.Lock()

# --- optional: trained future-lens (AO + naive) via LoRA adapters + Karvonen injection ---
# Loaded only if AO_CKPT is set. Injects a norm-matched activation at block 1 and GENERATES
# the readout, so it sits next to the linear lenses. The base forward in /api/run is wrapped
# in PEFT.disable_adapter() so the cached activations for the linear lenses stay clean.
AO_CKPT = os.environ.get("AO_CKPT", "")
NAIVE_CKPT = os.environ.get("NAIVE_CKPT", "")
RL_CKPT = os.environ.get("RL_CKPT", "")   # naive future-lens RL-autoencoded on penultimate (GRPO recon reward)
PEFT = None
if AO_CKPT:
    from peft import PeftModel
    from nla.utils.hooks import register_karvonen_hook
    from nla.schema import compute_canonical_neighbors
    from nla.datagen.injection_tokens import find_injection_token
    ACTOR_TEMPLATE = ("You are shown an internal activation vector captured from a language model "
        "as it reads a passage of text. The vector, enclosed in <concept> tags, is taken at one "
        "position and encodes what the model is about to generate next. Output the text the model "
        "most likely produces immediately after this point.\n\n<concept>{injection_char}</concept>")
    PEFT = PeftModel.from_pretrained(model, AO_CKPT, adapter_name="ao").eval()
    if NAIVE_CKPT:
        PEFT.load_adapter(NAIVE_CKPT, adapter_name="naive")
    if RL_CKPT:
        PEFT.load_adapter(RL_CKPT, adapter_name="rl")
    _inj_char, _inj_id = find_injection_token(tok)
    _left, _right = compute_canonical_neighbors(tok, ACTOR_TEMPLATE, _inj_char, _inj_id)
    _vref = [None]; register_karvonen_hook(PEFT, _vref, _inj_id, _left, _right); PEFT._fl_vref = _vref
    _s = tok.apply_chat_template([{"role": "user", "content": ACTOR_TEMPLATE.format(injection_char=_inj_char)}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
    _PT = torch.tensor([tok.encode(_s, add_special_tokens=False)], device=dev)
    print(f"[wc] future-lens adapters loaded: ao{' + naive' if NAIVE_CKPT else ''}{' + rl' if RL_CKPT else ''}", flush=True)


@torch.no_grad()
def _brollout(activation, adapter, n=4, max_new=24, temp=0.7):
    """Inject a norm-matched activation and batch-generate n rollouts with the given adapter."""
    act = torch.as_tensor(activation, dtype=torch.float32, device=dev).view(1, -1)
    B = max(1, n); ids = _PT.repeat(B, 1)
    PEFT.set_adapter(adapter); PEFT._fl_vref[0] = act.expand(B, -1).contiguous()
    try:
        # FIXED-LENGTH generation (min==max): this DeltaNet/fla build recompiles the generate
        # kernel per generated sequence length, so variable-length rollouts trigger a ~90s
        # compile on every new length (the "endless loading" bug). Forcing exactly max_new tokens
        # makes the shape constant -> compiles ONCE (at warmup) -> every click is fast. Cost: a
        # rollout may run a few tokens past a natural stop; fine for a short lens preview.
        g = PEFT.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                          min_new_tokens=max_new, max_new_tokens=max_new,
                          do_sample=(temp > 0), temperature=max(temp, 1e-5), top_p=0.95,
                          pad_token_id=tok.eos_token_id)
    finally:
        PEFT._fl_vref[0] = None
    return [tok.decode(x[_PT.shape[1]:], skip_special_tokens=True).strip() for x in g]


def unit_rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def rms_gain(x):
    return unit_rms(x.float()) * GAIN


@torch.no_grad()
def readout(h, head, use_j, layer, topk=10):
    """h (T, d) raw residual at `layer`. Returns top-k ids + probs per position."""
    z = h.to(torch.bfloat16)
    if use_j and layer in JMATS:
        z = z @ JMATS[layer].T
    z = z.float()
    if head == "lm":
        hidden = z
        scale = 1.0
    else:
        A, b, scale, xmode = HEADW[int(head)]
        x = unit_rms(z) if xmode == "unit" else z
        hidden = x @ A.T + b
    logits = (rms_gain(hidden).to(torch.bfloat16) @ W_U.T).float() * scale
    p = torch.softmax(logits, -1)
    v, i = p.topk(topk, -1)
    return i.cpu().tolist(), v.cpu().tolist()


class RunReq(BaseModel):
    """Either a real WeirdChat transcript (user + assistant, teacher-forced) or free
    text with optional generation."""
    user: str = ""
    assistant: str = ""
    system: str = ""          # optional system turn (e.g. the agentic-misalignment scenario)
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


app = FastAPI()


@app.get("/api/presets")
def presets(_=Depends(require_auth)):
    return JSONResponse({"behaviors": PRESET_JSON, "layers": JLAYERS,
                         "target": TARGET_L, "heads": sorted(HEADW)})


@app.post("/api/run")
@torch.no_grad()
def run(r: RunReq, _=Depends(require_auth)):
    def _ids(x):
        # transformers 5.x returns a BatchEncoding (a UserDict, NOT a dict subclass),
        # so test for the tensor case instead of testing for dict.
        return x if isinstance(x, torch.Tensor) else x["input_ids"]

    sysmsg = [{"role": "system", "content": r.system}] if r.system else []   # optional system turn
    if r.assistant:
        # real transcript: prompt = chat-templated (system+)user turn, then the model's OWN reply
        pre = _ids(tok.apply_chat_template(sysmsg + [{"role": "user", "content": r.user}],
                                           add_generation_prompt=True, return_tensors="pt"))
        rep = tok(r.assistant, add_special_tokens=False, return_tensors="pt").input_ids
        ids = torch.cat([pre, rep], 1).to(dev)
        n_prompt = pre.shape[1]
        gen = ids                       # nothing to sample: the reply is given
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
    if PEFT is not None:             # keep cached activations LoRA-free for the linear lenses
        with PEFT.disable_adapter():
            model(gen)
    else:
        model(gen)                   # one teacher-forced pass over prompt+continuation
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
    ids, ps = readout(run["acts"][r.layer], r.head, r.use_j, r.layer, r.topk)
    return {"tokens": run["tokens"], "n_prompt": run["n_prompt"],
            "top": [[[tok.decode([t]), round(p, 4)] for t, p in zip(a, b)]
                    for a, b in zip(ids, ps)]}


class AllReq(BaseModel):
    run_id: str
    pos: int
    layer: int
    use_j: bool = True
    topk: int = 10        # the UI asks for a deeper list when de-duplicating


@app.post("/api/allheads")
def allheads(r: AllReq, _=Depends(require_auth)):
    """Every head at ONE position, so the horizons can be compared side by side."""
    with LOCK:
        run = RUNS.get(r.run_id)
    if not run:
        raise HTTPException(404, "run expired -- reload the transcript")
    h = run["acts"][r.layer][r.pos:r.pos + 1]
    k = max(1, min(int(r.topk), 50))
    rows = []
    for head in ["lm"] + [str(k) for k in sorted(HEADW)]:
        i, p = readout(h, head, r.use_j, r.layer, k)
        rows.append({"head": head,
                     "top": [[tok.decode([t]), round(v, 4)] for t, v in zip(i[0], p[0])]})
    toks = run["tokens"]
    return {"pos": r.pos, "rows": rows,
            "actual_next": toks[r.pos + 1:r.pos + 9],
            "token": toks[r.pos],
            "is_assistant": r.pos >= run["n_prompt"]}


class AoReq(BaseModel):
    run_id: str
    pos: int
    layer: int
    use_j: bool = True
    n: int = 1        # batched DeltaNet decode is pathologically slow for n>1 in this build; keep 1
    max_new: int = 20


@app.post("/api/ao")
def api_ao(r: AoReq, _=Depends(require_auth)):
    """Trained future-lens (AO + naive) rollouts from the activation at (layer, pos).
    use_j feeds J_{layer->62}(h) (block-62 basis); else the raw activation."""
    if PEFT is None:
        raise HTTPException(404, "future-lens adapters not loaded (set AO_CKPT)")
    with LOCK:
        run = RUNS.get(r.run_id)
    if not run:
        raise HTTPException(404, "run expired -- reload the transcript")
    h = run["acts"][r.layer][r.pos].float()
    if r.use_j and r.layer in JMATS:
        h = JMATS[r.layer].float() @ h        # -> block-62 basis (the frame the AO was trained near)
    n, mx = max(1, min(int(r.n), 6)), max(4, min(int(r.max_new), 48))
    with LOCK:                       # injection hook state (_fl_vref/input_ids) is shared -> serialize
        out = {"pos": r.pos, "layer": r.layer, "use_j": r.use_j, "ao": _brollout(h, "ao", n, mx)}
        if NAIVE_CKPT:
            out["naive"] = _brollout(h, "naive", n, mx)
        if RL_CKPT:
            out["rl"] = _brollout(h, "rl", n, mx)
    return out


@app.get("/", response_class=HTMLResponse)
def index(_=Depends(require_auth)):
    # read from disk each time: the model takes ~10 min to load, so the UI must be
    # editable without restarting the server
    return open(UI_FILE).read()


UI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "weirdchat_ui.html")


if __name__ == "__main__":
    if PEFT is not None:                 # pre-compile EACH adapter's generate shape at the UI's exact
        # n/max_new so the first click of every lens is fast (the DeltaNet fla kernel recompiles per
        # (adapter, generate-length); the UI matrix always calls n=1, max_new=14).
        _z = torch.zeros(W_U.shape[1], device=dev)
        for _ad in [a for a in ("ao", "naive", "rl")
                    if (a == "ao") or (a == "naive" and NAIVE_CKPT) or (a == "rl" and RL_CKPT)]:
            try:
                _brollout(_z, _ad, n=1, max_new=14)
                print(f"[wc] warmup: {_ad} generate shape compiled (n=1,max_new=14)", flush=True)
            except Exception as e:
                print(f"[wc] warmup {_ad} skipped: {e!r}", flush=True)
    print(f"[wc] serving on 0.0.0.0:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
