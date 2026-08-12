"""DECISIVE CONTROL: feed arm A its in-distribution input (stored LOCAL JVP
transports) through the EVAL code path (multislot_fed_eval.py machinery) and
check it reproduces the known 8-token rollout.

If local transports reproduce the rollout through this path -> eval path is
CORRECT and the reported degradation is a property of the averaged Jbar
inputs. If local transports ALSO give filler -> eval-path wiring bug.

Everything below that builds prompts / hooks / injection is copied VERBATIM
from evals/multislot_fed_eval.py (same template, same vref tiling, same
adapter loading); additions are observers + metrics only.

Run on the GPU box:
  source /workspace/venv/bin/activate && source /workspace/.keys.env
  export PYTHONPATH=/workspace/skip-lens:/workspace/skip-lens/evals
  CUDA_VISIBLE_DEVICES=<free> python /workspace/skip-lens/diag_evalpath_control.py
"""
import glob
import hashlib
import json
import os
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from fl_common import base_causal
from nla.datagen.injection_tokens import find_injection_token
from nla.schema import compute_canonical_neighbors, INJECT_PLACEHOLDER
from nla.utils.hooks import register_karvonen_hook
from nla.train_sft import _av_prepare_chunk  # training-path probe

AV_CKPT = "/workspace/skip-lens/ckpts/multislot_armA_k8/iter_0003875"
BASE_CKPT = "Qwen/Qwen3.6-27B"
JBAR_DIR = "/workspace/results/offset_jlens"
SHARD = "/workspace/data/spans_jvp/shard_3_jvp.parquet"
VAL_PARQUET = "/workspace/data/final/jvp_multislot_val.parquet"
EVALS_DIR = "/workspace/skip-lens/evals/datasets/official/evaluations"
OUT_JSON = "/workspace/skip-lens/diag_out/evalpath_control.json"
N_ITEMS = 64
K = 8
SRC_LAYER, TGT_LAYER = 42, 62
dev = "cuda"
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
REPORT = {}

# ---------------------------------------------------------------------------
# Model + hooks: VERBATIM the multislot_fed_eval.py setup
# ---------------------------------------------------------------------------
ACTOR_TEMPLATE_MULTI = (
    "You are shown {k} internal activation vectors captured from a language "
    "model as it reads a passage of text. The vectors, enclosed in <concept> "
    "tags, are the model's state at one position transported to {k} "
    "consecutive future positions: the first vector encodes what the model is "
    "about to generate next, the second what it will generate after that, and "
    "so on. Output the text the model most likely produces over these {k} "
    "positions.\n\n<concept>{markers}</concept>")

tok = AutoTokenizer.from_pretrained(BASE_CKPT)
base = AutoModelForCausalLM.from_pretrained(
    BASE_CKPT, torch_dtype=torch.bfloat16,
    attn_implementation="sdpa").to(dev).eval()
model = PeftModel.from_pretrained(base, AV_CKPT, adapter_name="msA").eval()
inj_char, inj_id = find_injection_token(tok)
TEMPLATE = ACTOR_TEMPLATE_MULTI.format(k=K, markers="{injection_char}" * K)
left, right = compute_canonical_neighbors(tok, TEMPLATE, inj_char, inj_id)

# observers around the karvonen hook on decoder layer 1 (registration order
# matters: pre-observer BEFORE register_karvonen_hook, post-observer AFTER)
from nla.utils.arch_adapters import resolve_decoder_layers
_layer1 = resolve_decoder_layers(model.get_base_model())[1]
_obs = {"pre": None, "post": None, "on": False}


def _pre_hook(m, a, o):
    if _obs["on"]:
        _obs["pre"] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()


def _post_hook(m, a, o):
    if _obs["on"]:
        _obs["post"] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()


_layer1.register_forward_hook(_pre_hook)
vref = [None]
register_karvonen_hook(model, vref, inj_id, left, right)
_layer1.register_forward_hook(_post_hook)
torch.set_grad_enabled(False)

print(f"[diag] inj_char={inj_char!r} inj_id={inj_id} left={left} right={right}")
try:
    _aa = model.active_adapters
    _aa = _aa() if callable(_aa) else _aa
except Exception:
    _aa = [getattr(model, "active_adapter", "?")]
n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora" in n)
print(f"[diag] active adapters: {_aa} | lora params: {n_lora/1e6:.1f}M")
REPORT["setup"] = {"inj_char": inj_char, "inj_id": inj_id, "left": left,
                   "right": right, "active_adapters": list(_aa),
                   "lora_params": n_lora}

Jbar = {}
for d in range(K):
    p = os.path.join(JBAR_DIR, f"Jbar_L{SRC_LAYER}_to_L{TGT_LAYER}_off{d}.npy")
    Jbar[d] = torch.from_numpy(np.load(p)).float().to(dev)
Jpool = torch.from_numpy(np.load(os.path.join(
    JBAR_DIR, f"Jbar_L{SRC_LAYER}_to_L{TGT_LAYER}_offpooled.npy"))).float().to(dev)

grab = {}
base_causal(model).model.layers[SRC_LAYER].register_forward_hook(
    lambda m, i, o: grab.__setitem__(
        SRC_LAYER, (o[0] if isinstance(o, tuple) else o).detach()))

_prompt_cache = {}


def prompt_ids():
    if "ids" not in _prompt_cache:
        content = TEMPLATE.format(injection_char=inj_char)
        s = tok.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        _prompt_cache["ids"] = torch.tensor(
            [tok.encode(s, add_special_tokens=False)], dtype=torch.long,
            device=dev)
        _prompt_cache["str"] = s
    return _prompt_cache["ids"]


def brollout_slots(slots, n, max_new, temp=0.7):
    """VERBATIM copy of multislot_fed_eval.brollout_slots (+ returns ids)."""
    pt = prompt_ids()
    B = max(1, n)
    ids = pt.repeat(B, 1)
    vref[0] = slots.float().repeat(B, 1).contiguous()  # [B*K, d], row-major
    try:
        g = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new, do_sample=(temp > 0),
            temperature=max(temp, 1e-5), top_p=0.95,
            pad_token_id=tok.eos_token_id)
    finally:
        vref[0] = None
    txt = [tok.decode(x[pt.shape[1]:], skip_special_tokens=True).strip()
           for x in g]
    return txt, g[:, pt.shape[1]:].cpu()


def brollout_batch(slot_list, max_new):
    """Batched greedy variant: one prompt row per ITEM, vref = concat of each
    item's [K,d] block -> [B*K, d] row-major, identical injection semantics."""
    pt = prompt_ids()
    B = len(slot_list)
    ids = pt.repeat(B, 1)
    vref[0] = torch.cat([s.float() for s in slot_list], 0).contiguous()
    try:
        g = model.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=tok.eos_token_id)
    finally:
        vref[0] = None
    txt = [tok.decode(x[pt.shape[1]:], skip_special_tokens=True).strip()
           for x in g]
    return txt, g[:, pt.shape[1]:].cpu()


# ---------------------------------------------------------------------------
# 0) Template parity: training-style prompt construction == eval prompt_ids
# ---------------------------------------------------------------------------
train_content = ACTOR_TEMPLATE_MULTI.format(
    k=K, markers=INJECT_PLACEHOLDER * K).replace(INJECT_PLACEHOLDER, inj_char)
train_str = tok.apply_chat_template(
    [{"role": "user", "content": train_content}], tokenize=False,
    add_generation_prompt=True, enable_thinking=False)
eval_str = (prompt_ids(), _prompt_cache["str"])[1]
train_ids = tok.encode(train_str, add_special_tokens=False)
eval_ids_l = prompt_ids()[0].tolist()
tmpl_match = (train_str == eval_str) and (train_ids == eval_ids_l)
print(f"[0] template parity train==eval: str={train_str == eval_str} "
      f"ids={train_ids == eval_ids_l} n_markers={eval_ids_l.count(inj_id)}")
REPORT["template_parity"] = {
    "str_equal": train_str == eval_str, "ids_equal": train_ids == eval_ids_l,
    "n_markers_in_prompt": eval_ids_l.count(inj_id),
    "prompt_len": len(eval_ids_l)}
if not tmpl_match:
    print("[0] *** TEMPLATE MISMATCH between train and eval construction ***")
    import difflib
    for l in difflib.unified_diff(train_str.splitlines(), eval_str.splitlines(),
                                  lineterm=""):
        print("    " + l)

# ---------------------------------------------------------------------------
# 1) Load 64 held-out rows from shard_3 (val doc-hash split, finalize filters)
# ---------------------------------------------------------------------------
def is_val(doc_id, val_frac=0.03):
    h = int(hashlib.md5(str(doc_id).encode()).hexdigest(), 16) % 10000
    return h < int(val_frac * 10000)


rows = []
pf = pq.ParquetFile(SHARD)
for rg in range(pf.num_row_groups):
    for r in pf.read_row_group(rg).to_pylist():
        if not is_val(r["doc_id"]):
            continue
        if r["h42_recompute_cosine"] < 0.99 or len(r["rollout_token_ids"]) < K:
            continue
        resp = tok.decode(r["rollout_token_ids"][:K], skip_special_tokens=True)
        if len(resp.strip()) < 2:
            continue
        tv = np.frombuffer(r["transported_vectors"], dtype=np.float16)
        tv = tv.reshape(16, -1).astype(np.float32)
        rows.append({
            "doc_id": r["doc_id"], "ctx_text": r["ctx_text"],
            "roll": [int(x) for x in r["rollout_token_ids"]],
            "resp": resp,
            "tv": torch.from_numpy(tv.copy()),        # [16, d]
            "h42": torch.tensor(r["activation_vector"], dtype=torch.float32),
        })
        if len(rows) >= N_ITEMS:
            break
    if len(rows) >= N_ITEMS:
        break
print(f"[1] loaded {len(rows)} held-out rows from {SHARD}")
d_model = rows[0]["tv"].shape[1]

# cross-check vs the finalized val parquet ((doc_id, ctx_text) join):
# pass 1 = light columns for row indices, pass 2 = take() the heavy column
want = {(r["doc_id"], r["ctx_text"]): r for r in rows}
light = pq.read_table(VAL_PARQUET, columns=["doc_id", "ctx_text", "response"])
idx_map = {}
for i, (di, cx) in enumerate(zip(light.column("doc_id").to_pylist(),
                                 light.column("ctx_text").to_pylist())):
    if (di, cx) in want and (di, cx) not in idx_map:
        idx_map[(di, cx)] = i
n_checked = n_vec_ok = n_resp_ok = 0
if idx_map:
    keys = list(idx_map)
    heavy = pq.read_table(VAL_PARQUET, columns=["activation_vector"]).take(
        [idx_map[k] for k in keys])
    resp_col = light.column("response").to_pylist()
    for k, av in zip(keys, heavy.column("activation_vector")):
        r = want[k]
        va = np.array(av.as_py(), dtype=np.float32)
        mine = r["tv"][:K].reshape(-1).numpy()
        if va.shape == mine.shape and np.array_equal(va, mine):
            n_vec_ok += 1
        n_resp_ok += int(resp_col[idx_map[k]] == r["resp"])
        n_checked += 1
print(f"[1] val-parquet cross-check: {n_checked} matched, "
      f"vec identical {n_vec_ok}, resp identical {n_resp_ok}")
REPORT["val_crosscheck"] = {"matched": n_checked, "vec_identical": n_vec_ok,
                           "resp_identical": n_resp_ok}

# ---------------------------------------------------------------------------
# 2) Slot->marker ordering probe (runtime, both paths)
# ---------------------------------------------------------------------------
def marker_positions(ids_row):
    return [i for i, t in enumerate(ids_row.tolist()) if t == inj_id]


def identify(delta, dims):
    """Which probe dim dominates this marker's injection delta."""
    vals = {d: abs(float(delta[d])) for d in dims}
    best = max(vals, key=vals.get)
    return best, vals[best] / (float(delta.norm()) + 1e-9)


B_probe = 3
dims = [[1000 * b + 137 * k + 11 for k in range(K)] for b in range(B_probe)]
all_dims = [d for row in dims for d in row]

# eval path: same slots per batch row (that's all the eval ever does)
slots_probe = torch.zeros(K, d_model)
for k in range(K):
    slots_probe[k, dims[0][k]] = 1.0
pt = prompt_ids()
ids = pt.repeat(B_probe, 1)
_obs["on"] = True
vref[0] = slots_probe.float().repeat(B_probe, 1).contiguous()
model(input_ids=ids, attention_mask=torch.ones_like(ids))
vref[0] = None
_obs["on"] = False
delta = _obs["post"] - _obs["pre"]
mpos = marker_positions(ids[0])
eval_map = []
for b in range(B_probe):
    for k, p in enumerate(mpos):
        got, frac = identify(delta[b, p], dims[0])
        eval_map.append({"batch": b, "marker": k, "got_dim": got,
                         "want_dim": dims[0][k], "purity": round(frac, 4),
                         "ok": got == dims[0][k]})
ok_eval = all(e["ok"] for e in eval_map)
print(f"[2] eval-path slot->marker order: {'OK' if ok_eval else 'BROKEN'} "
      f"({sum(e['ok'] for e in eval_map)}/{len(eval_map)})")

# training path: DIFFERENT vectors per example through _av_prepare_chunk
probe_rows = []
for b in range(B_probe):
    flat = torch.zeros(K, d_model)
    for k in range(K):
        flat[k, dims[b][k]] = 1.0
    probe_rows.append({
        "prompt": [{"role": "user", "content": ACTOR_TEMPLATE_MULTI.format(
            k=K, markers=INJECT_PLACEHOLDER * K)}],
        "response": "probe text",
        "activation_vector": flat.reshape(-1).tolist(),
    })
t_ids, t_attn, t_mask, v_batch = _av_prepare_chunk(
    probe_rows, tok, inj_char, dev, n_slots=K)
_obs["on"] = True
vref[0] = v_batch
model(input_ids=t_ids, attention_mask=t_attn)
vref[0] = None
_obs["on"] = False
delta_t = _obs["post"] - _obs["pre"]
train_map = []
for b in range(B_probe):
    for k, p in enumerate(marker_positions(t_ids[b])):
        got, frac = identify(delta_t[b, p], all_dims)
        train_map.append({"ex": b, "marker": k, "got_dim": got,
                          "want_dim": dims[b][k], "purity": round(frac, 4),
                          "ok": got == dims[b][k]})
ok_train = all(e["ok"] for e in train_map)
print(f"[2] train-path slot->marker order: {'OK' if ok_train else 'BROKEN'} "
      f"({sum(e['ok'] for e in train_map)}/{len(train_map)})")
REPORT["ordering_probe"] = {"eval_ok": ok_eval, "train_ok": ok_train,
                            "eval_map": eval_map, "train_map": train_map}

# ---------------------------------------------------------------------------
# 3) Norm-matching: per-slot norms + runtime invariance to per-slot scaling
# ---------------------------------------------------------------------------
tv_norms = torch.stack([r["tv"].norm(dim=-1) for r in rows])  # [N, 16]
mean_norms = tv_norms.mean(0)
print("[3] mean local-transport norm per slot:",
      [round(float(x), 2) for x in mean_norms])
print(f"[3] slot0/slot7 = {float(mean_norms[0]/mean_norms[7]):.1f}x, "
      f"slot0/slot15 = {float(mean_norms[0]/mean_norms[15]):.1f}x")

r0 = rows[0]
s_base = r0["tv"][:K].clone()
scale = torch.tensor([2.0 ** d for d in range(K)]).unsqueeze(1)
outs = {}
for tag, s in [("base", s_base), ("scaled", s_base * scale)]:
    vref[0] = s.float().to(dev)
    _obs["on"] = True
    o = model(input_ids=pt, attention_mask=torch.ones_like(pt))
    _obs["on"] = False
    vref[0] = None
    outs[tag] = {"logits": o.logits[0, -1].float().cpu(),
                 "delta": (_obs["post"] - _obs["pre"])[0],
                 "pre": _obs["pre"][0]}
logit_diff = (outs["base"]["logits"] - outs["scaled"]["logits"]).abs().max()
dn = outs["base"]["delta"][mpos].norm(dim=-1)
hn = outs["base"]["pre"][mpos].norm(dim=-1)
print(f"[3] max|logit diff| slots vs 2^d-scaled slots: {float(logit_diff):.3e}")
print("[3] ||delta||/||h_p|| at markers:",
      [round(float(x), 4) for x in (dn / hn)])
REPORT["norm_matching"] = {
    "mean_slot_norms_16": [float(x) for x in mean_norms],
    "ratio_slot0_slot7": float(mean_norms[0] / mean_norms[7]),
    "ratio_slot0_slot15": float(mean_norms[0] / mean_norms[15]),
    "max_logit_diff_under_2powd_scaling": float(logit_diff),
    "delta_over_h_norm_at_markers": [float(x) for x in (dn / hn)]}

# ---------------------------------------------------------------------------
# 4) Jbar slot construction stats: orientation + collinearity
# ---------------------------------------------------------------------------
def pairwise_cos(S):
    Sn = S / (S.norm(dim=-1, keepdim=True) + 1e-9)
    C = Sn @ Sn.T
    iu = torch.triu_indices(K, K, offset=1)
    return float(C[iu[0], iu[1]].mean())


cos_fwd, cos_T, pc_local, pc_jbar, jn, tn = [], [], [], [], [], []
for r in rows:
    h = r["h42"].to(dev)
    per = torch.stack([Jbar[d] @ h for d in range(K)])
    perT = torch.stack([Jbar[d].T @ h for d in range(K)])
    tv8 = r["tv"][:K].to(dev)
    cos_fwd.append(float(torch.nn.functional.cosine_similarity(per, tv8, dim=-1).mean()))
    cos_T.append(float(torch.nn.functional.cosine_similarity(perT, tv8, dim=-1).mean()))
    pc_local.append(pairwise_cos(tv8))
    pc_jbar.append(pairwise_cos(per))
    jn.append(per.norm(dim=-1).cpu())
    tn.append(tv8.norm(dim=-1).cpu())
print(f"[4] mean cos(Jbar@h42, local tv): {np.mean(cos_fwd):.4f} | "
      f"transposed: {np.mean(cos_T):.4f}")
print(f"[4] mean pairwise slot cos: local {np.mean(pc_local):.3f}, "
      f"Jbar {np.mean(pc_jbar):.3f}")
REPORT["jbar_stats"] = {
    "cos_jbar_vs_local": float(np.mean(cos_fwd)),
    "cos_jbarT_vs_local": float(np.mean(cos_T)),
    "pairwise_cos_local": float(np.mean(pc_local)),
    "pairwise_cos_jbar": float(np.mean(pc_jbar)),
    "jbar_slot_norms": torch.stack(jn).mean(0).tolist(),
    "local_slot_norms": torch.stack(tn).mean(0).tolist()}

# ---------------------------------------------------------------------------
# 5) THE CONTROL: generate through the eval path, all conditions, greedy
# ---------------------------------------------------------------------------
DERANGE = [(i + K // 2) % K for i in range(K)]  # eval's shuffled_slots perm


def slots_for_row(cond, r):
    if cond == "local":
        return r["tv"][:K].to(dev)
    if cond == "local_shuffled":
        return r["tv"][:K][DERANGE].to(dev)
    h = r["h42"].to(dev)
    if cond == "jbar_per_offset":
        return torch.stack([Jbar[d] @ h for d in range(K)])
    if cond == "jbar_transposed":
        return torch.stack([Jbar[d].T @ h for d in range(K)])
    if cond == "jbar_pooled":
        return (Jpool @ h).expand(K, -1).contiguous()
    raise ValueError(cond)


CONDS = ["local", "local_shuffled", "jbar_per_offset", "jbar_transposed",
         "jbar_pooled"]
MAX_NEW = 16
gen = {c: [] for c in CONDS}
BS = 16
model.set_adapter("msA")  # mirror the real eval loop
for c in CONDS:
    for i in range(0, len(rows), BS):
        batch = rows[i:i + BS]
        txt, gids = brollout_batch([slots_for_row(c, r) for r in batch], MAX_NEW)
        for r, t, g in zip(batch, txt, gids):
            gen[c].append({"txt": t, "ids": g.tolist()})
    print(f"[5] generated {c}: {len(gen[c])} items")

# faithfulness spot-check: verbatim single-item brollout_slots (greedy) must
# equal the batched output
mismatch = 0
for r, gbatch in zip(rows[:4], gen["local"][:4]):
    txt, gids = brollout_slots(slots_for_row("local", r), 1, MAX_NEW, temp=0.0)
    if gids[0].tolist() != gbatch["ids"]:
        mismatch += 1
        print(f"[5] SPOT-CHECK DIFF:\n  single: {txt[0][:80]!r}\n"
              f"  batch : {gbatch['txt'][:80]!r}")
print(f"[5] verbatim-brollout_slots vs batched: {4 - mismatch}/4 identical")
REPORT["spotcheck_single_vs_batch"] = {"n": 4, "identical": 4 - mismatch}


def metrics(cond):
    ft, pfx_resp, pfx_roll, ov, ex, sp = [], [], [], [], [], []
    for r, g in zip(rows, gen[cond]):
        gids = g["ids"]
        roll8 = r["roll"][:K]
        resp_ids = tok.encode(r["resp"], add_special_tokens=False)
        def npfx(a, b):
            n = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                n += 1
            return n
        first_ok = bool(gids) and (gids[0] == roll8[0] or
                                   (bool(resp_ids) and gids[0] == resp_ids[0]))
        ft.append(int(first_ok))
        pfx_roll.append(npfx(gids, roll8))
        pfx_resp.append(npfx(gids, resp_ids))
        c1, c2 = Counter(gids[:K]), Counter(roll8)
        ov.append(sum((c1 & c2).values()) / K)
        rs = r["resp"].strip()
        sp.append(int(g["txt"].startswith(rs[:min(len(rs), 12)])))
        ex.append({"true": r["resp"], "gen": g["txt"][:120]})
    return {
        "first_token_match": float(np.mean(ft)),
        "mean_exact_prefix_vs_rollout_ids": float(np.mean(pfx_roll)),
        "mean_exact_prefix_vs_reencoded_resp": float(np.mean(pfx_resp)),
        "median_exact_prefix_vs_reencoded_resp": float(np.median(pfx_resp)),
        "mean_token_overlap_frac": float(np.mean(ov)),
        "str_prefix12_match": float(np.mean(sp)),
        "examples": ex[:10],
    }


REPORT["control"] = {}
for c in CONDS:
    m = metrics(c)
    REPORT["control"][c] = m
    print(f"[5] {c:18s} first-tok {m['first_token_match']:.3f} | "
          f"prefix(resp) {m['mean_exact_prefix_vs_reencoded_resp']:.2f} | "
          f"prefix(roll) {m['mean_exact_prefix_vs_rollout_ids']:.2f} | "
          f"overlap {m['mean_token_overlap_frac']:.3f}")

# teacher-forced CE through the EVAL prompt construction (bridge to the
# reported heldout CE 0.920 that was computed with the training code)
def eval_path_ce(cond):
    tot, ntok = 0.0, 0
    for i in range(0, len(rows), 8):
        batch = rows[i:i + 8]
        seqs, plens = [], []
        for r in batch:
            resp_ids = tok.encode(r["resp"] + (tok.eos_token or ""),
                                  add_special_tokens=False)
            seqs.append(eval_ids_l + resp_ids)
            plens.append(len(eval_ids_l))
        T = max(len(s) for s in seqs)
        ids_b = torch.full((len(seqs), T), tok.eos_token_id, dtype=torch.long,
                           device=dev)
        attn = torch.zeros((len(seqs), T), dtype=torch.long, device=dev)
        lmask = torch.zeros((len(seqs), T), dtype=torch.float32, device=dev)
        for j, s in enumerate(seqs):
            ids_b[j, :len(s)] = torch.tensor(s, device=dev)
            attn[j, :len(s)] = 1
            lmask[j, plens[j]:len(s)] = 1
        vref[0] = torch.cat([slots_for_row(cond, r).float() for r in batch], 0)
        try:
            logits = model(input_ids=ids_b, attention_mask=attn).logits.float()
        finally:
            vref[0] = None
        sl = logits[:, :-1]
        st = ids_b[:, 1:]
        sm = lmask[:, 1:]
        per = torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.shape[-1]), st.reshape(-1),
            reduction="none").view(st.shape)
        tot += float((per * sm).sum())
        ntok += int(sm.sum())
    return tot / max(ntok, 1)


REPORT["eval_path_ce"] = {}
for c in ["local", "jbar_per_offset", "jbar_pooled"]:
    ce = eval_path_ce(c)
    REPORT["eval_path_ce"][c] = ce
    print(f"[5] eval-path teacher-forced CE ({c}): {ce:.4f}  "
          f"(training-code heldout reference: 0.920)")

# ---------------------------------------------------------------------------
# 6) h42 norm distribution: training corpus rows vs eval items (eval grab)
# ---------------------------------------------------------------------------
GEN_PAD = 256
PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def grab_inputs(ids):
    real = ids[:, -GEN_PAD:]
    n = real.shape[1]
    rp = GEN_PAD - n
    if rp <= 0:
        return real, torch.ones_like(real), n - 1
    pad = torch.full((1, rp), PAD_ID, device=real.device, dtype=real.dtype)
    mask = torch.cat([torch.ones(1, n, dtype=torch.long, device=real.device),
                      torch.zeros(1, rp, dtype=torch.long, device=real.device)], 1)
    return torch.cat([real, pad], 1), mask, n - 1


items = []
for f in sorted(glob.glob(os.path.join(EVALS_DIR, "lens-eval-*.json"))):
    for it in json.load(open(f))["items"]:
        items.append(it["prompt"])
items = items[:64]
eval_norms = []
for p in items:
    ids_p = tok(p, return_tensors="pt").input_ids.to(dev)
    gids, gmask, last = grab_inputs(ids_p)
    with model.disable_adapter():
        model(input_ids=gids, attention_mask=gmask)
    eval_norms.append(float(grab[SRC_LAYER][0, last].float().norm()))
train_norms = [float(r["h42"].norm()) for r in rows]
print(f"[6] h42 norms: train corpus mean {np.mean(train_norms):.1f} "
      f"(p10 {np.percentile(train_norms,10):.1f}, p90 {np.percentile(train_norms,90):.1f}) | "
      f"eval items mean {np.mean(eval_norms):.1f} "
      f"(p10 {np.percentile(eval_norms,10):.1f}, p90 {np.percentile(eval_norms,90):.1f})")
REPORT["h42_norms"] = {
    "train_mean": float(np.mean(train_norms)),
    "train_p10_p90": [float(np.percentile(train_norms, q)) for q in (10, 90)],
    "eval_mean": float(np.mean(eval_norms)),
    "eval_p10_p90": [float(np.percentile(eval_norms, q)) for q in (10, 90)],
}

json.dump(REPORT, open(OUT_JSON, "w"), ensure_ascii=False, indent=1)
print(f"[done] wrote {OUT_JSON}")
