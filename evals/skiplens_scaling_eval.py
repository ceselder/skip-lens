"""SKIP-LENS SCALING EVAL — one AV checkpoint in, four skip-lens metrics out.

Answers "does more futurelens pretraining -> better skip-lens performance?" by
sweeping FED LAYERS (feed the L62-trained AV the raw residual from shallower depths,
norm-matched at block 1) and reporting, per fed layer:

  1. fve          — feed h_l, AV reads it, frozen AR reconstructs, FVE vs the TRUE L62 activation.
  2. coherence    — Sonnet judge (nla.utils.text_judges) on the readouts.
  3. a6_discrim   — A.6 intermediate-surfacing discrimination (evals/judge_intermediates) on readouts
                    generated from the paper's evals/datasets items at each fed layer.
  4. greedy_lp    — mean teacher-forced logprob of the model's OWN greedy continuation of the
                    context, under [ACTOR_TEMPLATE + inject h_l]. High at L62 (normal), degrades
                    toward shallow layers (skip) if the lens is faithful.

Run once per pretraining checkpoint; a driver (run_skiplens_scaling.sh) sweeps the 10 checkpoints
and scripts/plot_skiplens_scaling.py turns the JSONs into the scaling curves.
"""
import argparse, glob, json, math, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from nla.utils.hooks import register_karvonen_hook
from nla.schema import (compute_canonical_neighbors, extract_explanation,
                        normalize_activation, compute_predict_mean_baselines,
                        resolve_target_scale)
from nla.datagen.injection_tokens import find_injection_token
from nla.utils.arch_adapters import resolve_text_model
from nla.utils.critic import critic_predict
from nla.config import load_nla_config
from nla.utils.text_judges import judge_explanations
from nla.train_sft import init_critic_from_base
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file as _load_file
from pathlib import Path
import json as _json

# --- inlined fl_common helpers (self-contained so this runs on any box) -------------
ACTOR_TEMPLATE = (
    "You are shown an internal activation vector captured from a language model "
    "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
    "taken at one position and encodes what the model is about to generate next. "
    "Output the text the model most likely produces immediately after this point.\n\n"
    "<concept>{injection_char}</concept>"
)


def base_causal(model):
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return resolve_text_model(inner)


def prompt_ids(tok, inj_char, device):
    content = ACTOR_TEMPLATE.format(injection_char=inj_char)
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return torch.tensor([tok.encode(s, add_special_tokens=False)], dtype=torch.long, device=device)


# ------------------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--av-ckpt", required=True, help="futurelens AV LoRA (the checkpoint under test)")
ap.add_argument("--ar-ckpt", required=True, help="frozen AR/critic (NLACriticModel dir)")
ap.add_argument("--sidecar", required=True, help="parquet whose .nla_meta.yaml carries mse_scale/inj/d_model + critic template")
ap.add_argument("--ctx-parquet", required=True, help="held-out parquet with a `ctx_text` column (fed-layer FVE/coherence/greedy set)")
ap.add_argument("--a6-dir", default=None, help="evals/datasets dir for A.6 surfacing (skip if unset)")
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--fed-layers", default="62,55,48,42,34,26,18,10")
ap.add_argument("--n-fve", type=int, default=128, help="held-out ctx items for FVE/coherence/greedy")
ap.add_argument("--n-a6", type=int, default=40, help="items per A.6 distribution")
ap.add_argument("--n-ao", type=int, default=1, help="readout samples per item (1=greedy)")
ap.add_argument("--rollout-len", type=int, default=48)
ap.add_argument("--greedy-cont-len", type=int, default=24, help="length of the model's greedy continuation for greedy_lp")
ap.add_argument("--ce-samples", type=int, default=4, help="temp-1 sampled continuations per ctx for the CE metric (seeded => identical across checkpoints, so H_model is a clean shared floor)")
ap.add_argument("--ce-cont-len", type=int, default=24, help="length of each sampled continuation for the CE metric")
ap.add_argument("--tag", default=None)
ap.add_argument("--out", required=True)
args = ap.parse_args()
dev = "cuda"
FED = [int(x) for x in args.fed_layers.split(",")]
L62 = max(FED) if 62 not in FED else 62
GRAB = sorted(set(FED + [62]))
tag = args.tag or os.path.basename(args.av_ckpt.rstrip("/"))

# --- load model + AV + AR + config --------------------------------------------------
tok = AutoTokenizer.from_pretrained(args.base_ckpt)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
base = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
model = PeftModel.from_pretrained(base, args.av_ckpt, adapter_name="AV").eval()
inj_char, inj_id = find_injection_token(tok)
left, right = compute_canonical_neighbors(tok, ACTOR_TEMPLATE, inj_char, inj_id)
vref = [None]
register_karvonen_hook(model, vref, inj_id, left, right)
model._fl_vref = vref
torch.set_grad_enabled(False)

cfg = load_nla_config(args.sidecar, tok)
mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
critic_template = cfg.critic_prompt_template
assert critic_template is not None, "critic_prompt_template missing from sidecar meta"
# AR is an AR-LoRA + value_head (base + injected LoRA loaded from a state dict) — replicate
# train_rl_self_contained.py's exact rebuild (init_critic_from_base -> inject LoRA -> load sd).
ar_meta = _json.loads((Path(args.ar_ckpt) / "ar_meta.json").read_text())
critic = init_critic_from_base(
    args.base_ckpt, ar_meta["ar_num_layers"], torch.bfloat16, None,
    device_map=None, max_memory=None,
    strip_final_norm=ar_meta.get("final_norm_stripped", True)).to(dev).eval()
inject_adapter_in_model(LoraConfig(
    r=ar_meta["lora_r"], lora_alpha=ar_meta["lora_alpha"], lora_dropout=0.0,
    bias="none", task_type="CAUSAL_LM", use_rslora=True,
    target_modules=ar_meta["target_modules"]), critic.backbone)
_ar_sd = _load_file(str(Path(args.ar_ckpt) / "ar_lora_value_head.safetensors"))
_miss, _unexp = critic.load_state_dict(_ar_sd, strict=False)
assert sum(1 for k in _ar_sd if "lora_" in k) > 0 and not _unexp, f"AR load mismatch: unexpected={_unexp[:3]}"
for p_ in critic.parameters():
    p_.requires_grad_(False)
critic.eval()
print(f"[cfg] inj_id={inj_id} mse_scale_f={mse_scale_f:.3f} d_model={cfg.d_model} fed={FED} tag={tag} | AR: {len(_ar_sd)} tensors loaded", flush=True)

# per-layer residual grab (output of decoder block l), populated by a clean adapter-OFF forward
grab = {}
layers = base_causal(model).model.layers
for l in GRAB:
    layers[l].register_forward_hook(
        lambda m, i, o, L=l: grab.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))


@torch.no_grad()
def capture_layers(text):
    ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev)
    with model.disable_adapter():
        model(input_ids=ids)                         # populate grab (adapter off = clean base residuals)
    last = ids.shape[1] - 1
    return {l: grab[l][0, last].float().clone() for l in GRAB}, ids


@torch.no_grad()
def av_readout(h_l, n=1, temp=0.0, max_new=None):
    max_new = max_new or args.rollout_len
    pt = prompt_ids(tok, inj_char, dev)
    B = max(1, n)
    model._fl_vref[0] = h_l.view(1, -1).expand(B, -1).contiguous()
    try:
        g = model.generate(input_ids=pt.repeat(B, 1), attention_mask=torch.ones(B, pt.shape[1], device=dev, dtype=torch.long),
                           max_new_tokens=max_new, do_sample=(temp > 0),
                           **({"temperature": temp} if temp > 0 else {}), top_p=1.0 if temp == 0 else 0.95,
                           pad_token_id=tok.eos_token_id)
    finally:
        model._fl_vref[0] = None
    return [tok.decode(x[pt.shape[1]:], skip_special_tokens=True).strip() for x in g]


@torch.no_grad()
def greedy_continuation(ctx_ids, max_new):
    with model.disable_adapter():
        g = model.generate(input_ids=ctx_ids, attention_mask=torch.ones_like(ctx_ids),
                           max_new_tokens=max_new, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    return g[0, ctx_ids.shape[1]:]                   # continuation token ids


@torch.no_grad()
def logprob_of_continuation(h_l, cont_ids):
    """mean teacher-forced logprob of cont_ids under [ACTOR_TEMPLATE + inject h_l]."""
    pt = prompt_ids(tok, inj_char, dev)
    if cont_ids.numel() == 0:
        return float("nan")
    seq = torch.cat([pt, cont_ids.view(1, -1)], dim=1)
    model._fl_vref[0] = h_l.view(1, -1)
    try:
        logits = model(input_ids=seq, attention_mask=torch.ones_like(seq)).logits[0].float()
    finally:
        model._fl_vref[0] = None
    P = pt.shape[1]
    pred = logits[P - 1: P - 1 + cont_ids.shape[0]]  # position P-1+k predicts cont token k
    lp = pred.log_softmax(-1).gather(-1, cont_ids.view(-1, 1)).squeeze(-1)
    return float(lp.mean())


def _trunc_eos(ids):
    """drop everything from the first EOS on (so CE isn't measured over padding)."""
    nz = (ids == tok.eos_token_id).nonzero()
    return ids[: int(nz[0])] if nz.numel() else ids


@torch.no_grad()
def sampled_continuations(ctx_ids, k, max_new, temp=1.0, seed=0):
    """k temp-sampled continuations from the BASE model. Seeded => the SAME continuations
    are drawn for a given ctx across every checkpoint, so H_model is a clean shared floor and
    CE_L is measured on identical target tokens run-to-run."""
    torch.manual_seed(seed)
    with model.disable_adapter():
        g = model.generate(input_ids=ctx_ids.repeat(k, 1),
                           attention_mask=torch.ones(k, ctx_ids.shape[1], device=dev, dtype=torch.long),
                           max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=1.0,
                           pad_token_id=tok.eos_token_id)
    outs = [_trunc_eos(g[i, ctx_ids.shape[1]:]) for i in range(k)]
    return [c for c in outs if c.numel() > 0]


@torch.no_grad()
def base_ce_of_continuation(ctx_ids, cont_ids):
    """mean teacher-forced CE (nats) of cont_ids under the BASE model given the raw context =
    the model's self cross-entropy on its own continuation = the H_model entropy floor."""
    if cont_ids.numel() == 0:
        return float("nan")
    seq = torch.cat([ctx_ids, cont_ids.view(1, -1)], dim=1)
    with model.disable_adapter():
        logits = model(input_ids=seq, attention_mask=torch.ones_like(seq)).logits[0].float()
    C = ctx_ids.shape[1]
    pred = logits[C - 1: C - 1 + cont_ids.shape[0]]     # position C-1+k predicts cont token k
    lp = pred.log_softmax(-1).gather(-1, cont_ids.view(-1, 1)).squeeze(-1)
    return float(-lp.mean())


def fve_of_readout(readout, gold_L62):
    expl = extract_explanation(readout)
    if not expl:
        return None
    cids = tok.encode(critic_template.format(explanation=expl), add_special_tokens=False)
    if not (0 < len(cids) <= 1024):
        return None
    x = torch.tensor([cids], dtype=torch.long, device=dev)
    pred = critic_predict(critic, x, None, mse_scale_f)[0]
    pn = normalize_activation(pred.unsqueeze(0), mse_scale_f)[0]
    gn = normalize_activation(gold_L62.to(dev).unsqueeze(0), mse_scale_f)[0]
    mse = F.mse_loss(pn, gn).item()
    return mse if math.isfinite(mse) else None


# ==== METRICS 1,2,4 on the held-out ctx set =========================================
import pyarrow.parquet as pq
ctx_tbl = pq.read_table(args.ctx_parquet)
ctx_col = "ctx_text" if "ctx_text" in ctx_tbl.column_names else "detokenized_text_truncated"
ctx_rows = ctx_tbl.column(ctx_col).to_pylist()[: args.n_fve]
print(f"[ctx] {len(ctx_rows)} held-out items from {args.ctx_parquet}::{ctx_col}", flush=True)

per = {l: {"mse": [], "readouts": [], "greedy_lp": [], "ce": []} for l in FED}
h_model_ce = []            # base model's self-CE on its own sampled continuations = entropy floor
gold_L62_all = []
for j, text in enumerate(ctx_rows):
    if not text or not text.strip():
        continue
    h, ctx_ids = capture_layers(text)
    gold_L62_all.append(h[62].cpu())
    cont = greedy_continuation(ctx_ids, args.greedy_cont_len)
    # CE metric: temp-1 sampled continuations (seeded per ctx => identical across checkpoints).
    ce_conts = sampled_continuations(ctx_ids, args.ce_samples, args.ce_cont_len,
                                     temp=1.0, seed=1000 + j)
    h_model_ce.extend(base_ce_of_continuation(ctx_ids, c) for c in ce_conts)   # floor
    for l in FED:
        r = av_readout(h[l], n=args.n_ao, temp=0.0)[0]
        per[l]["readouts"].append(r)
        m = fve_of_readout(r, h[62])
        if m is not None:
            per[l]["mse"].append(m)
        per[l]["greedy_lp"].append(logprob_of_continuation(h[l], cont))
        # CE_L = -mean log p_lens(model's sampled continuation tokens), averaged over the K samples
        ces = [-logprob_of_continuation(h[l], c) for c in ce_conts]
        if ces:
            per[l]["ce"].append(float(np.mean(ces)))
    if j % 16 == 0:
        print(f"[ctx] {j}/{len(ctx_rows)}", flush=True)

# Baseline over a LARGE stable sample of stored L62 activations — n_fve alone (esp. small) badly
# underestimates the predict-the-mean baseline and flips FVE negative. Fall back to captured golds.
if "activation_vector" in ctx_tbl.column_names:
    _base = torch.tensor(ctx_tbl.column("activation_vector").to_pylist()[:512], dtype=torch.float32)
else:
    _base = torch.stack(gold_L62_all)
_, fve_baseline = compute_predict_mean_baselines(_base, mse_scale_f)
print(f"[fve] baseline mse_nrm = {fve_baseline:.4f} (over {_base.shape[0]} stored L62 acts)", flush=True)

# coherence: judge all readouts per fed layer (source = the readout itself; we only use coherence)
# H_model = the model's continuation entropy floor (nats): base model self-CE on its own temp-1 samples.
_hm = [x for x in h_model_ce if math.isfinite(x)]
h_model = float(np.mean(_hm)) if _hm else float("nan")
metrics = {"tag": tag, "av_ckpt": args.av_ckpt, "fed_layers": FED,
           "fve_baseline": float(fve_baseline),
           "h_model_ce": h_model, "n_ce_floor": len(_hm), "per_layer": {}}
print(f"[ce] H_model (model continuation entropy floor) = {h_model:.3f} nats "
      f"(over {len(_hm)} sampled continuations)", flush=True)
for l in FED:
    mses = per[l]["mse"]
    fve = (1.0 - float(np.mean(mses)) / fve_baseline) if mses else float("nan")
    lps = [x for x in per[l]["greedy_lp"] if math.isfinite(x)]
    ces = [x for x in per[l]["ce"] if math.isfinite(x)]
    ce = float(np.mean(ces)) if ces else float("nan")
    reads = per[l]["readouts"]
    try:
        jm, _ = judge_explanations(reads, reads, seed=0, concurrency=32)
        coh = jm.get("coherence_mean", float("nan"))
    except Exception as e:
        print(f"[judge] coherence failed L{l}: {e}", flush=True); coh = float("nan")
    metrics["per_layer"][l] = {
        "fve": fve, "fve_pct": fve * 100.0,
        "coherence": coh,
        "greedy_lp": float(np.mean(lps)) if lps else float("nan"),
        "ce": ce,                          # CE_L: -mean log p_lens(model's sampled continuation)
        "ce_gap": (ce - h_model) if math.isfinite(ce) else float("nan"),  # excess nats = KL(model||lens)
        "n_fve": len(mses), "n_ce": len(ces), "extraction_rate": len(mses) / max(1, len(reads)),
    }
    print(f"[L{l}] CE {ce:.3f} | gap {metrics['per_layer'][l]['ce_gap']:+.3f} nats | "
          f"FVE {fve*100:.1f}% | coherence {coh:.2f} | n={len(ces)}", flush=True)

# ==== METRIC 3: A.6 intermediate-surfacing (readout GENERATION only) ==================
# Decoupled from scoring: we write one readouts JSON per fed layer in the exact format
# evals/judge_intermediates.py expects ({dist: {item_idx: [bullets]}}); the driver then runs
#   python -m evals.judge_intermediates --readouts <f> --tag L{l}
# and plot_skiplens_scaling.py reads auc_discrimination from those score files.
if args.a6_dir:
    a6_dir_out = os.path.join(os.path.dirname(os.path.abspath(args.out)), f"a6_readouts_{tag}")
    os.makedirs(a6_dir_out, exist_ok=True)
    a6 = {l: {} for l in FED}              # {fed_layer: {dist: {idx: [bullets]}}}
    for f in sorted(glob.glob(os.path.join(args.a6_dir, "*.json"))):
        name = os.path.basename(f)[:-5]
        if name == "manifest":
            continue
        data = json.load(open(f))
        items = (data.get("items") if isinstance(data, dict) else data)[: args.n_a6]
        for l in FED:
            a6[l][name] = {}
        for i, it in enumerate(items):
            prompt = it["prompt"] if isinstance(it, dict) else it
            h, _ = capture_layers(prompt)
            for l in FED:
                a6[l][name][str(i)] = av_readout(h[l], n=4, temp=0.8, max_new=args.rollout_len)
        print(f"[a6] {name}: {len(items)} items x {len(FED)} fed-layers done", flush=True)
    a6_files = {}
    for l in FED:
        pth = os.path.join(a6_dir_out, f"readouts_L{l}.json")
        json.dump(a6[l], open(pth, "w"), ensure_ascii=False)
        a6_files[l] = pth
    metrics["a6_readout_files"] = a6_files
    print(f"[a6] wrote {len(a6_files)} readout files to {a6_dir_out} (score with judge_intermediates)", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
json.dump(metrics, open(args.out, "w"), ensure_ascii=False, indent=1)
print(f"[done] wrote {args.out}", flush=True)
