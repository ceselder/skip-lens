"""WeirdChat multi-token lens playground.

Browse Transluce/WeirdChat prompts (or paste your own), run Qwen3.6-27B, then pick
INDEPENDENTLY:

  * the JACOBIAN   J_{L->62}  for any source layer L that has a matrix (or "none")
  * the LM HEAD    the model's own W_U, or a fitted horizon-k head A_k

and read the lens out at every position. When ``BITTER_CKPT`` is provided,
layer 42 also exposes the learned activation-conditioned Bitter transport next
to the ordinary mean J-lens:

    lens(h_L) = softmax( W_U . norm( A_k . transport(h_L) ) )

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
    BASE, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}
).eval()
tm = resolve_text_model(model)
W_U = model.lm_head.weight.detach()
GAIN = norm_gain(model).to(dev)
EPS = float(getattr(tm.norm, "variance_epsilon", getattr(tm.norm, "eps", 1e-6)))
print(f"[wc] model ready. d={W_U.shape[1]} V={W_U.shape[0]}", flush=True)

JMATS = {L: torch.from_numpy(np.load(os.path.join(JDIR, f"J_L{L}_to_L{TARGET_L}.npy"))
                             ).to(dev).to(torch.bfloat16) for L in JLAYERS}

# Optional activation-conditioned L42->L62 transport. The direct checkpoint was
# trained on RMS-normalized source states, so its matched mean-J control is
# Jbar @ unit_rms(h), not the paper/raw Jbar @ h arm. Both are exposed in the UI.
BITTER_CKPT = os.environ.get("BITTER_CKPT", "")
BITTER_CODE = os.environ.get("BITTER_CODE", "/workspace-vast/celeste/bitter-lens")
BITTER = None
BITTER_LAYER = None
BITTER_META = {}
if BITTER_CKPT:
    if BITTER_CODE not in sys.path:
        sys.path.insert(0, BITTER_CODE)
    from bitter_lens import load_transport
    _payload = torch.load(BITTER_CKPT, map_location="cpu", weights_only=True)
    BITTER_LAYER = int(_payload["config"]["source_layer"])
    if BITTER_LAYER not in JMATS:
        raise RuntimeError(f"Bitter source L{BITTER_LAYER} has no mean J matrix in {JDIR}")
    BITTER, BITTER_META = load_transport(
        BITTER_CKPT, JMATS[BITTER_LAYER].float(), map_location=dev
    )
    BITTER = BITTER.to(dev).eval()
    print(f"[wc] Bitter Lens L{BITTER_LAYER}->{TARGET_L} loaded from {BITTER_CKPT} "
          f"(step={BITTER_META.get('step', '?')})", flush=True)
# R-lens transport matrices (RelP LRP rules). Same shape/use as J; feed R@h to skip-lens.
RDIR = os.environ.get("RDIR", "")
RMATS = {}
if RDIR:
    for _p in glob.glob(os.path.join(RDIR, f"R_L*_to_L{TARGET_L}.npy")):
        _L = int(re.search(rf"R_L(\d+)_to_L{TARGET_L}", os.path.basename(_p)).group(1))
        RMATS[_L] = torch.from_numpy(np.load(_p)).to(dev).to(torch.bfloat16)
    print(f"[wc] R-lens source layers {sorted(RMATS)} -> block {TARGET_L} (RDIR={RDIR})", flush=True)

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
REPEAT_CKPT = os.environ.get("REPEAT_CKPT", "")   # repeat-after-me lens adapter
REPEAT25K_CKPT = os.environ.get("REPEAT25K_CKPT", "")   # repeat-after-me, 25k / all-modules LoRA
CNLA_LH_CKPT = os.environ.get("CNLA_LH_CKPT", "")   # compositional-NLA long-horizon RL @ step 300 (default cNLA lens)
L42M_CKPT = os.environ.get("L42M_CKPT", "")   # skip-lens L42-matched adapter
L62MM_CKPT = os.environ.get("L62MM_CKPT", "")   # skip-lens L62-mismatch adapter (trained L62, fed L42)
# selectable on-policy cNLA checkpoints for the playground dropdown (lazy-loaded)
CNLA_CKPT_ROOTS = os.environ.get("CNLA_CKPT_ROOTS",
    "/workspace/cnla/skip-lens/ckpts/cnla_av_L62_big,/workspace/cnla/skip-lens/ckpts/cnla_longhorizon").split(",")
CNLA_CKPTS = {}     # display-name -> iter dir
PEFT = None
# --- generalized lazy adapter machinery (any path -> adapter name) ---
_ADAPTER_BY_PATH = {}   # normpath -> adapter_name; pre-populated with the named env adapters
_PROTECTED = set()      # named adapter names that must NEVER be LRU-evicted
_dyn_loaded = []        # LRU of lazily-loaded "dyn::" adapter names (cap 6)
REGISTRY = {}           # category label -> {ckpt label -> adapter dir}; served by /api/registry


def _norm_path(p):
    return os.path.normpath(str(p).rstrip("/")) if p else p


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
    if REPEAT_CKPT:
        PEFT.load_adapter(REPEAT_CKPT, adapter_name="repeat")
    if REPEAT25K_CKPT:
        PEFT.load_adapter(REPEAT25K_CKPT, adapter_name="repeat25k")
    if CNLA_LH_CKPT:
        PEFT.load_adapter(CNLA_LH_CKPT, adapter_name="cnla_lh")
    if L42M_CKPT:
        PEFT.load_adapter(L42M_CKPT, adapter_name="l42m")
    if L62MM_CKPT:
        PEFT.load_adapter(L62MM_CKPT, adapter_name="l62mm")
    import glob as _glob
    for _root in CNLA_CKPT_ROOTS:
        _root = _root.strip()
        _tag = ("warmstart" if _root.endswith("cnla_av_L62_big")
                else "longhorizon" if _root.endswith("cnla_longhorizon")
                else os.path.basename(_root))
        for _d in sorted(_glob.glob(_root + "/iter_*")):
            CNLA_CKPTS[f"{_tag}:{os.path.basename(_d)}"] = _d
    print(f"[wc] selectable cNLA checkpoints: {len(CNLA_CKPTS)}", flush=True)
    _inj_char, _inj_id = find_injection_token(tok)
    _left, _right = compute_canonical_neighbors(tok, ACTOR_TEMPLATE, _inj_char, _inj_id)
    _vref = [None]; register_karvonen_hook(PEFT, _vref, _inj_id, _left, _right); PEFT._fl_vref = _vref
    _s = tok.apply_chat_template([{"role": "user", "content": ACTOR_TEMPLATE.format(injection_char=_inj_char)}],
                                 tokenize=False, add_generation_prompt=True, enable_thinking=False)
    _PT = torch.tensor([tok.encode(_s, add_special_tokens=False)], device=dev)
    print(f"[wc] future-lens adapters loaded: ao{' + naive' if NAIVE_CKPT else ''}{' + rl' if RL_CKPT else ''}{' + repeat' if REPEAT_CKPT else ''}{' + repeat25k' if REPEAT25K_CKPT else ''}{' + cnla_lh' if CNLA_LH_CKPT else ''}{' + l42m' if L42M_CKPT else ''}{' + l62mm' if L62MM_CKPT else ''}", flush=True)
    # map each pre-loaded named adapter's path so _ensure_adapter reuses it (never re-loads)
    for _pth, _an in [(AO_CKPT, "ao"), (NAIVE_CKPT, "naive"), (RL_CKPT, "rl"),
                      (REPEAT_CKPT, "repeat"), (REPEAT25K_CKPT, "repeat25k"),
                      (CNLA_LH_CKPT, "cnla_lh"), (L42M_CKPT, "l42m"), (L62MM_CKPT, "l62mm")]:
        if _pth:
            _ADAPTER_BY_PATH[_norm_path(_pth)] = _an
            _PROTECTED.add(_an)

# --- comparison-slot registry: category -> {checkpoint label -> adapter dir} ---
REPEAT_SPAN4_CKPT = os.environ.get("REPEAT_SPAN4_CKPT",
                                   "/workspace/cnla/adapters/repeat_span4_25k_iter1500")
if AO_CKPT:
    REGISTRY["AO"] = {"default": AO_CKPT}
if NAIVE_CKPT:
    REGISTRY["naive-FL"] = {"default": NAIVE_CKPT}
if RL_CKPT:
    REGISTRY["FL-RL"] = {"default": RL_CKPT}
_rep = {lab: p for lab, p in [("main", REPEAT_CKPT), ("25k-allmodules", REPEAT25K_CKPT),
                              ("span4-25k-iter1500", REPEAT_SPAN4_CKPT)]
        if p and os.path.isdir(p)}
if _rep:
    REGISTRY["repeat"] = _rep
if L42M_CKPT:
    REGISTRY["L42-matched"] = {"default": L42M_CKPT}
if L62MM_CKPT:
    REGISTRY["L62-mismatch"] = {"default": L62MM_CKPT}
if CNLA_CKPTS:
    REGISTRY["cNLA"] = dict(sorted(CNLA_CKPTS.items()))
elif CNLA_LH_CKPT:
    REGISTRY["cNLA"] = {"default": CNLA_LH_CKPT}
if REGISTRY:
    print("[wc] registry: " + ", ".join(f"{c}({len(v)})" for c, v in REGISTRY.items()),
          flush=True)


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


def _ensure_adapter(path):
    """Resolve an adapter dir to a loaded adapter name.

    Pre-loaded named adapters (ao/naive/rl/repeat/...) are reused via _ADAPTER_BY_PATH and
    never evicted. Any other path is lazily loaded as "dyn::<sanitized>" with an LRU cap of 6
    (evicted via PEFT.delete_adapter) to bound GPU memory."""
    p = _norm_path(path)
    aname = _ADAPTER_BY_PATH.get(p)
    if aname is not None:
        if aname in _PROTECTED:
            return aname
        if aname in _dyn_loaded:              # refresh LRU position
            _dyn_loaded.remove(aname)
            _dyn_loaded.append(aname)
            return aname
    if not os.path.isdir(p):
        raise HTTPException(404, f"adapter dir not found: {p}")
    aname = "dyn::" + re.sub(r"[^A-Za-z0-9._-]+", "_", p).strip("_")
    PEFT.load_adapter(p, adapter_name=aname)
    _ADAPTER_BY_PATH[p] = aname
    _dyn_loaded.append(aname)
    while len(_dyn_loaded) > 6:               # only ever holds dyn adapters, never the named ones
        old = _dyn_loaded.pop(0)
        try:
            PEFT.delete_adapter(old)
        except Exception:
            pass
        for k in [k for k, v in _ADAPTER_BY_PATH.items() if v == old]:
            _ADAPTER_BY_PATH.pop(k, None)
    return aname


def _ensure_cnla_adapter(name):
    """Back-compat shim (legacy cnla_ckpt param): display name -> loaded adapter name."""
    if name not in CNLA_CKPTS:
        return "cnla_lh"
    return _ensure_adapter(CNLA_CKPTS[name])


def unit_rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def rms_gain(x):
    return unit_rms(x.float()) * GAIN


def _transport(h, layer, mode):
    """Map a raw source residual into the block-TARGET_L readout frame."""
    hf = h.float()
    if mode == "bitter":
        if BITTER is None or layer != BITTER_LAYER:
            raise HTTPException(404, f"Bitter Lens is only available at L{BITTER_LAYER}")
        return BITTER.transform(hf)
    if mode == "j_norm" and layer in JMATS:
        return unit_rms(hf) @ JMATS[layer].float().T
    if mode == "r" and layer in RMATS:
        return hf @ RMATS[layer].float().T
    if mode == "j" and layer in JMATS:
        return hf @ JMATS[layer].float().T
    return hf


@torch.no_grad()
def readout(h, head, use_j, layer, topk=10, feed=""):
    """h (T, d) raw residual at `layer`. Returns top-k ids + probs per position.
    feed selects the transport into the block-62 frame before the LM head."""
    mode = feed or ("j" if use_j else "raw")
    z = _transport(h, layer, mode)
    if head == "lm":
        hidden = z
        scale = 1.0
    else:
        if int(head) not in HEADW:
            raise HTTPException(404, f"horizon head k={head} not loaded (HEADS dir empty/missing)")
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
    feed: str = ""   # bitter / j_norm / j / r / raw; '' => use_j


app = FastAPI()


@app.get("/api/presets")
def presets(_=Depends(require_auth)):
    return JSONResponse({"behaviors": PRESET_JSON, "layers": JLAYERS,
                         "target": TARGET_L, "heads": sorted(HEADW),
                         "bitter_layers": [BITTER_LAYER] if BITTER is not None else [],
                         "bitter_meta": BITTER_META})


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
    ids, ps = readout(run["acts"][r.layer], r.head, r.use_j, r.layer, r.topk, feed=r.feed)
    return {"tokens": run["tokens"], "n_prompt": run["n_prompt"],
            "top": [[[tok.decode([t]), round(p, 4)] for t, p in zip(a, b)]
                    for a, b in zip(ids, ps)]}


class AllReq(BaseModel):
    run_id: str
    pos: int
    layer: int
    use_j: bool = True
    topk: int = 10        # the UI asks for a deeper list when de-duplicating
    feed: str = ""        # bitter / j_norm / j / r / raw; '' => use_j


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
        i, p = readout(h, head, r.use_j, r.layer, k, feed=r.feed)
        rows.append({"head": head,
                     "top": [[tok.decode([t]), round(v, 4)] for t, v in zip(i[0], p[0])]})
    toks = run["tokens"]
    return {"pos": r.pos, "rows": rows,
            "actual_next": toks[r.pos + 1:r.pos + 9],
            "token": toks[r.pos],
            "is_assistant": r.pos >= run["n_prompt"]}


class LensSlot(BaseModel):
    cat: str          # REGISTRY category label, e.g. "AO", "repeat", "cNLA"
    ckpt: str         # checkpoint label within that category, e.g. "default", "main"
    max_new: int = 0  # per-slot generated tokens (0 => default: 128 for cNLA, else the request mx)


class AoReq(BaseModel):
    run_id: str
    pos: int
    layer: int
    use_j: bool = True
    n: int = 1        # batched DeltaNet decode is pathologically slow for n>1 in this build; keep 1
    max_new: int = 20
    cnla_ckpt: str = ""   # LEGACY: pick which cNLA checkpoint generates the verbalization
    lenses: list[str] | None = None   # LEGACY: which lenses to run; None/empty => all available
    feed: str = ""        # pre-feed: bitter / j_norm / j / r / raw; "" => use_j
    slots: list[LensSlot] | None = None   # NEW: up to 3 (category, checkpoint) pairs to compare


@app.post("/api/ao")
def api_ao(r: AoReq, _=Depends(require_auth)):
    """Trained future-lens rollouts from the activation at (layer, pos).
    feed selects the pre-feed transform ("j"/"r"/"raw"; "" falls back to use_j).
    NEW: r.slots = up to 3 (category, checkpoint) pairs resolved via REGISTRY, one rollout each.
    LEGACY (r.slots is None): run the lenses named in r.lenses (empty => all)."""
    if PEFT is None:
        raise HTTPException(404, "future-lens adapters not loaded (set AO_CKPT)")
    with LOCK:
        run = RUNS.get(r.run_id)
    if not run:
        raise HTTPException(404, "run expired -- reload the transcript")
    h = run["acts"][r.layer][r.pos].float()
    feed = r.feed or ("j" if r.use_j else "raw")
    h = _transport(h, r.layer, feed)          # identical transform to the LM-head table
    n, mx = max(1, min(int(r.n), 6)), max(4, min(int(r.max_new), 48))
    if r.slots is not None:                        # ---- NEW slot-comparison path ----
        out_slots = []
        with LOCK:                   # injection hook state + adapter selection are shared -> serialize
            for s in r.slots[:3]:
                cell = {"cat": s.cat, "ckpt": s.ckpt}
                path = REGISTRY.get(s.cat, {}).get(s.ckpt)
                if path is None:
                    cell["readout"], cell["error"] = [], f"unknown slot {s.cat!r} / {s.ckpt!r}"
                    out_slots.append(cell)
                    continue
                try:
                    aname = _ensure_adapter(path)
                    # per-slot token budget: user's value if set, else default (cNLA needs ~128 for 4 bullets)
                    smx = (max(1, min(int(s.max_new), 256)) if s.max_new
                           else (max(mx, 128) if s.cat == "cNLA" else mx))
                    cell["readout"] = _brollout(h, aname, n, smx)
                    cell["max_new"] = smx
                except HTTPException as e:
                    cell["readout"], cell["error"] = [], str(e.detail)
                except Exception as e:
                    cell["readout"], cell["error"] = [], repr(e)[:200]
                out_slots.append(cell)
        return {"pos": r.pos, "layer": r.layer, "feed": feed, "slots": out_slots}
    # ---- LEGACY path (old checkbox UI; kept so cached pages don't crash) ----
    sel = set(r.lenses) if r.lenses else None      # None => run every available lens
    def want(k):
        return sel is None or k in sel
    with LOCK:                       # injection hook state (_fl_vref/input_ids) is shared -> serialize
        out = {"pos": r.pos, "layer": r.layer, "use_j": r.use_j}
        if want("ao"):
            out["ao"] = _brollout(h, "ao", n, mx)
        if NAIVE_CKPT and want("naive"):
            out["naive"] = _brollout(h, "naive", n, mx)
        if RL_CKPT and want("rl"):
            out["rl"] = _brollout(h, "rl", n, mx)
        if REPEAT_CKPT and want("repeat"):
            out["repeat"] = _brollout(h, "repeat", n, mx)
        if REPEAT25K_CKPT and want("repeat25k"):
            out["repeat25k"] = _brollout(h, "repeat25k", n, mx)
        if L42M_CKPT and want("l42m"):
            out["l42m"] = _brollout(h, "l42m", n, mx)
        if L62MM_CKPT and want("l62mm"):
            out["l62mm"] = _brollout(h, "l62mm", n, mx)
        if (CNLA_LH_CKPT or r.cnla_ckpt) and want("cnla_lh"):
            # cNLA emits 4 bullets — give it room (the shared mx=14 truncates to ~1 bullet).
            # cnla_ckpt lets the user pick which checkpoint verbalizes; else the wired default.
            _cad = _ensure_cnla_adapter(r.cnla_ckpt) if r.cnla_ckpt else "cnla_lh"
            out["cnla_lh"] = _brollout(h, _cad, n, max(mx, 128))
            out["cnla_ckpt"] = r.cnla_ckpt or "default (longhorizon:iter_000300)"
    return out


@app.get("/api/registry")
def api_registry(_=Depends(require_auth)):
    """Lens registry for the UI comparison-slot dropdowns: category -> [ckpt labels]."""
    return {"categories": {cat: list(ckpts.keys()) for cat, ckpts in REGISTRY.items()}}


@app.get("/api/cnla_ckpts")
def api_cnla_ckpts(_=Depends(require_auth)):
    """LEGACY: selectable on-policy cNLA checkpoints for the old playground dropdown."""
    return {"ckpts": sorted(CNLA_CKPTS.keys()),
            "default": "longhorizon:iter_000300 (wired)"}


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
        for _ad in [a for a in ("ao", "naive", "rl", "repeat", "repeat25k", "cnla_lh", "l42m", "l62mm")
                    if (a == "ao") or (a == "naive" and NAIVE_CKPT) or (a == "rl" and RL_CKPT)
                    or (a == "repeat" and REPEAT_CKPT) or (a == "repeat25k" and REPEAT25K_CKPT)
                    or (a == "cnla_lh" and CNLA_LH_CKPT) or (a == "l42m" and L42M_CKPT)
                    or (a == "l62mm" and L62MM_CKPT)]:
            try:
                _mn = 128 if _ad == "cnla_lh" else 14   # cNLA emits full bullets @128; compile that length
                _brollout(_z, _ad, n=1, max_new=_mn)
                print(f"[wc] warmup: {_ad} generate shape compiled (n=1,max_new={_mn})", flush=True)
            except Exception as e:
                print(f"[wc] warmup {_ad} skipped: {e!r}", flush=True)
    print(f"[wc] serving on 0.0.0.0:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
