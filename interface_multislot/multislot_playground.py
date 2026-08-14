"""Multi-slot workspace-lens playground (STANDALONE COPY — does not touch
interface/, which belongs to a sibling session).

Paste/run a prompt, click any token, and read it out through any trained decoder
with any test-time vector. The point of this version is that the VECTOR SOURCE and
the DECODER are independent dropdowns, so you can feed a vector to a decoder that
never trained on it — which is how the study's main result was found.

Measured on 551 official items, judged workspace agreement, normalised 0-1:

  DECODER          native vector                              score
  armI (K=1)       regression_avg, uniform d<16               0.539   <- best
  armC (K=1)       the paper's pooled J-lens vector           0.561 / 0.583 raw
  armG (K=8)       centered averaged slots                    0.565 (control pending)
  armE (K=8)       averaged 62->63 slots, read at 42->63      0.527 (control 0.027)
  armF (K=8)       averaged 42->63 slots, matched             0.517
  armD (K=8)       REAL penultimate states                    0.309 fed estimates
  armA (K=8)       LOCAL per-offset transports                0.144 fed averaged

Same decoder, different vector (arm I): regression 0.539 > raw h42 0.453 >
plain averaged Jacobian 0.342 > position-independent mean 0.018. So E[J] as the
paper computes it loses to just feeding the activation, and what rescues it is
fitting the averaged operator by regression instead:
W* = E[(Jh)h^T].E[hh^T]^-1, which is still ONE fixed corpus-level matrix.

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
REG_DIR = os.environ.get("REG_DIR", "/workspace/results/regression_lens")
CKPT_ROOT = os.environ.get("CKPT_ROOT", "/workspace/skip-lens/ckpts")
TARGET_L, SRC_L, K = 62, 42, 8
NOFF = 16                      # per-offset families are fit out to 16 horizons
PORT = int(os.environ.get("PORT", 8807))
dev = "cuda"


def _latest(d):
    """newest iter_* under a checkpoint dir, or None if the arm has not run"""
    if not os.path.isdir(d):
        return None
    its = sorted(x for x in os.listdir(d) if x.startswith("iter_"))
    return os.path.join(d, its[-1]) if its else None


print("[ms] loading model ...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
tm = resolve_text_model(model)
W_U = model.lm_head.weight.detach()
GAIN = norm_gain(model).to(dev)
EPS = float(getattr(tm.norm, "variance_epsilon", getattr(tm.norm, "eps", 1e-6)))

# ---------------------------------------------------------------- operators
JBAR = [torch.from_numpy(np.load(
    os.path.join(JBAR_DIR, f"Jbar_L{SRC_L}_to_L{TARGET_L}_off{d}.npy"))
    ).to(dev).float() for d in range(NOFF)]
JPOOL = torch.from_numpy(np.load(
    os.path.join(JBAR_DIR, f"Jbar_L{SRC_L}_to_L{TARGET_L}_offpooled.npy"))).to(dev).float()
JMATS = {SRC_L: JPOOL.to(torch.bfloat16)}
HBAR42 = torch.from_numpy(np.load(
    os.path.join(JBAR_DIR, f"hbar_L{SRC_L}.npy"))).float().to(dev)
MEANDIR = {}
for _d in range(NOFF):
    _p = os.path.join(JBAR_DIR, f"meandir_off{_d}.npy")
    if os.path.exists(_p):
        MEANDIR[_d] = torch.from_numpy(np.load(_p)).float().to(dev)

# The two poolings the arms were trained with. Weights are computed from the
# Jbar norms in BOTH cases, so a regression-pooled vector uses exactly the
# weights its arm trained on rather than its own (different) matrix norms.
W_UNIFORM = [1.0] * NOFF                                     # arm I
W_DEEP = [0.0] + [1.0 / float(JBAR[d].norm()) for d in range(1, NOFF)]   # arm K


def _pool_from_dir(directory, prefix, weights):
    """Weighted sum of a per-offset family, accumulated one matrix at a time so
    peak memory stays at one matrix rather than sixteen. Exact rather than an
    approximation: sum_d w_d W_d = (sum_d w_d B_d) A^-1, i.e. the same thing as
    fitting the pooled target directly."""
    acc = None
    for d in range(NOFF):
        if not weights[d]:
            continue
        p = os.path.join(directory, f"{prefix}_L{SRC_L}_to_L{TARGET_L}_off{d}.npy")
        if not os.path.exists(p):
            return None
        M = torch.from_numpy(np.load(p)).float().to(dev) * weights[d]
        acc = M if acc is None else acc + M
        del M
    return acc


OPS = {
    "jbar_uniform": _pool_from_dir(JBAR_DIR, "Jbar", W_UNIFORM),
    "jbar_deep": _pool_from_dir(JBAR_DIR, "Jbar", W_DEEP),
    "regr_uniform": _pool_from_dir(REG_DIR, "Wreg", W_UNIFORM),
    "regr_deep": _pool_from_dir(REG_DIR, "Wreg", W_DEEP),
    "regrU_uniform": _pool_from_dir(REG_DIR, "WregU", W_UNIFORM),
}
OPS = {k: v for k, v in OPS.items() if v is not None}
print("[ms] pooled operators: " + ", ".join(
    f"{k} |M|={float(v.norm()):.1f}" for k, v in OPS.items()), flush=True)

# Every entry is one fixed corpus-level matrix applied to h42 (or h42 itself),
# so none of them can see the test context's own gradient. Labels carry the
# measured score of the arm whose NATIVE vector this is.
VECSRC = {
    "regr_uniform": ("regression W*, pooled d<16  [armI 0.539]",
                     lambda h: OPS["regr_uniform"] @ h),
    # NOTE: Jbar_..._offpooled.npy is exactly (1/16) * sum_d Jbar_d — cosine
    # 1.0000, norm ratio 16.0 — and the injection hook norm-matches to ||h_p||,
    # so this is the SAME INPUT as jbar_uniform. Kept as a separate entry only
    # because armC's measured score was obtained under this name. Validated
    # against an independently-fit external J-lens at 0.905 matrix cosine.
    "jlens_pooled": ("the paper's pooled J-lens vector (identical input to "
                     "jbar_uniform after norm-matching)  [armC 0.561]",
                     lambda h: JPOOL @ h),
    "raw_h42": ("raw h42, no operator  [armI 0.453 / armK 0.517]",
                lambda h: h),
    "regr_deep": ("regression W*, deep-only norm-eq  [armK 0.423]",
                  lambda h: OPS["regr_deep"] @ h),
    "jbar_deep": ("averaged E[J], deep-only norm-eq  [armK 0.389]",
                  lambda h: OPS["jbar_deep"] @ h),
    "jbar_uniform": ("averaged E[J], pooled d<16  [armI 0.342]",
                     lambda h: OPS["jbar_uniform"] @ h),
    "regrU_uniform": ("regression W* fit on v/||v||, pooled d<16  [untested]",
                      lambda h: OPS["regrU_uniform"] @ h),
    "mean_only": ("position-independent mean  [FLOOR, armI 0.018]",
                  lambda h: OPS["regr_uniform"] @ HBAR42),
}
# mean_only is built from regr_uniform, so it survives only if that one did
VECSRC = {k: v for k, v in VECSRC.items()
          if k in ("raw_h42", "jlens_pooled") or k in OPS
          or (k == "mean_only" and "regr_uniform" in OPS)}
VEC_ORDER = [k for k in ("regr_uniform", "jlens_pooled", "raw_h42", "regr_deep",
                         "jbar_deep", "jbar_uniform", "regrU_uniform", "mean_only")
             if k in VECSRC]
JLAYERS, CAP_LAYERS = [SRC_L], [SRC_L, TARGET_L]

# Per-offset REGRESSION families, so the K=8 arms can be fed W* slots and not
# only plain-E[J] ones. Without these the multi-slot arms could never be tested
# with the operator that gave the single-slot arms their +0.197.
def _load_family(directory, prefix, k):
    out = []
    for d in range(k):
        p = os.path.join(directory, f"{prefix}_L{SRC_L}_to_L{TARGET_L}_off{d}.npy")
        if not os.path.exists(p):
            return None
        out.append(torch.from_numpy(np.load(p)).float().to(dev))
    return out


WREG = _load_family(REG_DIR, "Wreg", K)
WREGU = _load_family(REG_DIR, "WregU", K)
print(f"[ms] per-offset regression families: Wreg={'yes' if WREG else 'NO'} "
      f"WregU={'yes' if WREGU else 'NO'}", flush=True)

SUFFIX_DIR = os.path.join(JBAR_DIR, "suffix")
JSUF = None
if os.path.isdir(SUFFIX_DIR):
    try:
        JSUF = [torch.from_numpy(np.load(os.path.join(
            SUFFIX_DIR, f"Jbar_L{SRC_L}_to_L{TARGET_L}_off{d}.npy"))).to(dev).float()
            for d in range(K)]
    except Exception as e:
        print(f"[ms] suffix family unavailable: {e!r}", flush=True)

# ---------------------------------------------------------------- adapters
from peft import PeftModel
from nla.datagen.injection_tokens import find_injection_token
from nla.schema import compute_canonical_neighbors
from nla.utils.hooks import register_karvonen_hook
from pretrain.finalize_jvp_spans import ACTOR_TEMPLATE_MULTI

# arm C was trained with "about to generate next"; arms I and K with "over the
# next several tokens". Feeding an arm a template it never saw penalises it for
# the wrong reason, so each decoder keeps its own.
T_NEXT = ("You are shown an internal activation vector captured from a language model "
          "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
          "taken at one position and encodes what the model is about to generate next. "
          "Output the text the model most likely produces immediately after this point."
          "\n\n<concept>{injection_char}</concept>")
T_SPAN = ("You are shown an internal activation vector captured from a language model "
          "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
          "taken at one position and encodes what the model is about to generate over "
          "the next several tokens. Output the text the model most likely produces "
          "immediately after this point.\n\n<concept>{injection_char}</concept>")
MULTI_TEMPLATE = ACTOR_TEMPLATE_MULTI.format(k=K, markers="{injection_char}" * K)

# (category, adapter, ckpt dir, kind, template, default vector for single arms)
ARM_SPECS = [
    ("armI-1slot", "armI", f"{CKPT_ROOT}/armI_pooled_single", "single", T_SPAN,
     "regr_uniform"),
    ("armK-1slot", "armK", f"{CKPT_ROOT}/armK_deep_single", "single", T_SPAN,
     "regr_deep"),
    ("armC-1slot", "armC", f"{CKPT_ROOT}/multislot_armC_L62", "single", T_NEXT,
     "jlens_pooled"),
    ("armG-K8", "armG", f"{CKPT_ROOT}/multislot_armG_centered", "multi",
     MULTI_TEMPLATE, None),
    ("armE-K8", "armE", f"{CKPT_ROOT}/multislot_armE_twoJ", "multi",
     MULTI_TEMPLATE, None),
    ("armF-K8", "armF", f"{CKPT_ROOT}/multislot_armF_matched", "multi",
     MULTI_TEMPLATE, None),
    ("armD-K8", "armD", f"{CKPT_ROOT}/multislot_armD_pen8", "multi",
     MULTI_TEMPLATE, None),
    ("armA-K8", "armA", f"{CKPT_ROOT}/multislot_armA_k8", "multi",
     MULTI_TEMPLATE, None),
    ("armFrozen-K8", "armFrozen", f"{CKPT_ROOT}/multislot_armA_frozen_k8", "multi",
     MULTI_TEMPLATE, None),
    # K=3, so it needs its own 3-marker template and 3-slot constructions. Worst
    # arm measured (0.140, 85% degenerate) but included to complete the set.
    ("armH-K3", "armH", f"{CKPT_ROOT}/multislot_armH_pen3", "multi3",
     ACTOR_TEMPLATE_MULTI.format(k=3, markers="{injection_char}" * 3), None),
]

ARMS = {}          # category -> (adapter, kind, prompt_ids, default vector)
PEFT = None
inj_char, inj_id = find_injection_token(tok)


def _pt(template):
    s = tok.apply_chat_template(
        [{"role": "user", "content": template.format(injection_char=inj_char)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return torch.tensor([tok.encode(s, add_special_tokens=False)], device=dev)


for cat, aname, root, kind, tmpl, dflt in ARM_SPECS:
    ck = _latest(root)
    if ck is None:
        print(f"[ms] skip {cat}: no checkpoint under {root}", flush=True)
        continue
    if PEFT is None:
        PEFT = PeftModel.from_pretrained(model, ck, adapter_name=aname).eval()
    else:
        PEFT.load_adapter(ck, adapter_name=aname)
    ARMS[cat] = (aname, kind, _pt(tmpl), dflt)
    print(f"[ms] loaded {cat} <- {ck}", flush=True)
assert PEFT is not None, "no arm checkpoints found"

# canonical flanks are identical for a run of 1 and a run of K (same tag tokens)
_left, _right = compute_canonical_neighbors(tok, MULTI_TEMPLATE, inj_char, inj_id)
_vref = [None]
_sref = [None, 1]        # [slot_scale, n_slots] — see nla/utils/hooks.py
register_karvonen_hook(PEFT, _vref, inj_id, _left, _right, scale_ref=_sref)

_BASE_CONDS = ["per_offset", "centered", "diff", "keep0_deflate_rest",
               "keep0_gs_rest", "slot0_only", "shuffled_slots", "no_slot0", "gs",
               "pooled_identical", "deflated"]
_BASE_CONDS = [c for c in _BASE_CONDS if c in CONDITIONS]
# regression variants first for the two constructions worth comparing head to
# head; the rest of the E[J] constructions follow
COND_ORDER = []
if WREG is not None:
    COND_ORDER += ["regr_per_offset", "regr_centered", "regr_diff"]
if WREGU is not None:
    COND_ORDER += ["regrU_per_offset"]
COND_ORDER += _BASE_CONDS
if JSUF is not None:
    COND_ORDER += ["suffix_shared", "suffix_perslot"]
# single-slot arms expose every vector source, so any vector can be fed to any
# decoder — including ones that never trained on it
REGISTRY = {cat: (COND_ORDER if kind.startswith("multi") else VEC_ORDER)
            for cat, (_a, kind, _p, _d) in ARMS.items()}
VEC_LABELS = {k: VECSRC[k][0] for k in VEC_ORDER}

RUNS, LOCK = {}, threading.Lock()


def slots_for(cond, h42, k=K):
    """Delegates to evals/slot_builders.py — the same code path the judged evals
    use, so the playground can never drift from the measured numbers.

    A `regr_`/`regrU_` prefix swaps the per-offset FAMILY the slots are built
    from (ridge-fit W*_d instead of E[J]_d) while leaving the construction
    identical, so e.g. regr_centered is the centered construction on the
    regression family."""
    if cond in ("suffix_shared", "suffix_perslot"):
        if JSUF is None:
            raise HTTPException(404, "suffix-pooled family not built")
        slots = build_slots("per_offset", h42, JSUF, JPOOL, k=k)
        return slots, ("shared" if cond == "suffix_shared" else "per_slot")
    fam, base = JBAR[:k], cond
    for pfx, f in (("regrU_", WREGU), ("regr_", WREG)):
        if cond.startswith(pfx):
            if f is None:
                raise HTTPException(404, f"{pfx}* family not fitted")
            fam, base = f[:k], cond[len(pfx):]
            break
    # jpool stays the plain pooled J-lens: the constructions that use it
    # (pooled_identical) are defined against the paper's vector by design
    return build_slots(base, h42, fam, JPOOL, k=k, hbar=HBAR42,
                       meandirs=MEANDIR), "per_slot"


def vec_for(name, h42):
    if name not in VECSRC:
        raise HTTPException(404, f"unknown vector source {name!r}")
    return VECSRC[name][1](h42).unsqueeze(0)


@torch.no_grad()
def _roll(vectors, adapter, prompt_ids, n=1, max_new=14, temp=0.7,
          slot_scale="per_slot"):
    """vectors: [S, d] (S = K for the 8-slot arms, 1 for the single-slot arms)."""
    B = max(1, n)
    ids = prompt_ids.repeat(B, 1)
    PEFT.set_adapter(adapter)
    _sref[0], _sref[1] = slot_scale, vectors.shape[0]
    _vref[0] = vectors.float().repeat(B, 1).contiguous()
    try:
        g = PEFT.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                          min_new_tokens=max_new, max_new_tokens=max_new,
                          do_sample=(temp > 0), temperature=max(temp, 1e-5),
                          top_p=0.95, pad_token_id=tok.eos_token_id)
    finally:
        _vref[0] = None
        _sref[0] = None
    return [tok.decode(x[prompt_ids.shape[1]:], skip_special_tokens=True).strip()
            for x in g]


def unit_rms(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)


def apply_feed(h, name, layer):
    """h: [N, d] fp32 at `layer` -> the vector that goes into norm+unembed.

    `name` may be any VECSRC key, so the LM-head (logit-lens) readout can be
    taken through the regression operator and not only through E[J]. Worth
    knowing before reading the rows: measured on real continuations, the
    regression fit is WORSE than E[J] under this readout (+0.038 vs +0.057 span
    recall) because ridge minimises squared error on the transport while the
    unembedding sees only direction — but its offset-0 part is a much sharper
    next-token predictor (+0.349 vs +0.195). Sharper on the immediate token,
    weaker on the span."""
    if name in ("", "raw"):
        return h
    if name == "j":                      # legacy: the pooled J-lens
        return (JPOOL @ h.T).T if layer == SRC_L else h
    if name == "raw_h42":
        return h
    if name == "jlens_pooled":
        return (JPOOL @ h.T).T
    if name == "mean_only":
        return (OPS["regr_uniform"] @ HBAR42).unsqueeze(0).expand_as(h)
    if name in OPS:
        return (OPS[name] @ h.T).T
    raise HTTPException(404, f"unknown feed {name!r}")


@torch.no_grad()
def readout(h, head, use_j, layer, topk=10, feed=""):
    mode = feed or ("j" if use_j else "raw")
    hidden = apply_feed(h.float(), mode, layer)
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


EVALS_DIR = os.environ.get(
    "EVALS_DIR", "/workspace/skip-lens/evals/datasets/official/evaluations")
PRESETS_PATH = os.environ.get("PRESETS", "")


def _eval_behaviors():
    """The 551 official eval items, grouped by file, as browsable behaviours.

    chat=False because these are raw completion prompts — the judged evals feed
    them untemplated, and wrapping them in a chat turn would read out a different
    activation than the measured numbers correspond to."""
    out = {}
    for f in sorted(glob.glob(os.path.join(EVALS_DIR, "lens-eval-*.json"))):
        nm = os.path.basename(f)[len("lens-eval-"):-len(".json")]
        try:
            items = json.load(open(f)).get("items", [])
        except Exception as e:
            print(f"[ms] eval file {f} unreadable: {e!r}", flush=True)
            continue
        out[f"eval:{nm}"] = [
            {"user": it["prompt"], "chat": False,
             "explanation": f"official eval item · {nm} · {it.get('name', '')}"}
            for it in items if it.get("prompt")]
    return out


# The weirdchat preset file is passed through VERBATIM and its contents are never
# inspected here — the user asked that the agentic-misalignment scenarios not be
# read. Point PRESETS at the json (it lives on the SLURM box by default, so copy
# it over and set the env var) and its behaviours appear alongside the evals.
BEHAVIORS = _eval_behaviors()
if PRESETS_PATH and os.path.exists(PRESETS_PATH):
    try:
        _pj = json.load(open(PRESETS_PATH))
        BEHAVIORS = {**BEHAVIORS, **_pj}
        print(f"[ms] presets merged from {PRESETS_PATH}: "
              f"{len(_pj)} behaviour group(s), contents not inspected", flush=True)
    except Exception as e:
        print(f"[ms] presets at {PRESETS_PATH} unreadable: {e!r}", flush=True)
elif PRESETS_PATH:
    print(f"[ms] PRESETS={PRESETS_PATH} does not exist — evals only", flush=True)
print(f"[ms] behaviours: " + ", ".join(
    f"{k} ({len(v)})" for k, v in BEHAVIORS.items()), flush=True)


@app.get("/api/presets")
def presets(_=Depends(require_auth)):
    return JSONResponse({"behaviors": BEHAVIORS, "layers": JLAYERS,
                         "target": TARGET_L, "heads": []})


@app.get("/api/registry")
def api_registry(_=Depends(require_auth)):
    return {"categories": {cat: list(v) for cat, v in REGISTRY.items()},
            "vector_labels": VEC_LABELS}


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
    # One row per VECTOR SOURCE, all through the model's own norm + lm_head. This
    # is the untrained linear readout of each operator, so it is directly
    # comparable to the trained-decoder rollouts below it and shows what the
    # regression operator looks like without a decoder in the way.
    rows = []
    feeds = ([r.feed] if r.feed else
             (["raw_h42"] + [v for v in VEC_ORDER if v != "raw_h42"]))
    for feed in feeds:
        try:
            i, p = readout(h, "lm", r.use_j, layer, k, feed=feed)
            rows.append({"head": f"lm_head <- {feed}",
                         "top": [[tok.decode([t]), round(v, 4)]
                                 for t, v in zip(i[0], p[0])]})
        except HTTPException as e:
            rows.append({"head": f"lm_head <- {feed}", "top": [],
                         "error": str(e.detail)})
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
        LensSlot(cat="armI-1slot", ckpt="regr_uniform"),
        LensSlot(cat="armI-1slot", ckpt="jbar_uniform"),
        LensSlot(cat="armI-1slot", ckpt="raw_h42"),
    ]
    out_slots = []
    with LOCK:
        for s in slots_req[:3]:
            cell = {"cat": s.cat, "ckpt": s.ckpt}
            try:
                smx = max(1, min(int(s.max_new), 64)) if s.max_new else mx
                if s.cat not in ARMS:
                    raise HTTPException(404, f"unknown category {s.cat!r}")
                aname, kind, pt, dflt = ARMS[s.cat]
                if kind.startswith("multi"):
                    if s.ckpt not in COND_ORDER:
                        raise HTTPException(404, f"unknown condition {s.ckpt!r}")
                    kk = 3 if kind == "multi3" else K
                    slots, sscale = slots_for(s.ckpt, h42, k=kk)
                    cell["readout"] = _roll(slots, aname, pt, n, smx,
                                            slot_scale=sscale)
                    cell["slot_scale"] = sscale
                else:
                    src = s.ckpt or dflt
                    cell["readout"] = _roll(vec_for(src, h42), aname, pt, n, smx)
                    cell["vector"] = VECSRC[src][0]
                    cell["native"] = (src == dflt)
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
    for _cat, (_ad, _k, _pt_, _d) in ARMS.items():
        try:
            _S = 3 if _k == "multi3" else (K if _k == "multi" else 1)
            _roll(torch.zeros(_S, W_U.shape[1], device=dev), _ad, _pt_, n=1, max_new=8)
            print(f"[ms] warmup ok: {_cat}", flush=True)
        except Exception as e:
            print(f"[ms] warmup {_cat} skipped: {e!r}", flush=True)
    print(f"[ms] {len(ARMS)} decoders x {len(VEC_ORDER)} vector sources", flush=True)
    print(f"[ms] serving on 0.0.0.0:{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
