"""Pin the ae_L62 collection convention by probing what the ae-trained AR expects.

fl_big is confirmed standard (layers[62] output @ last ctx token). Recompute fl_big's
activations across a GRID of (layer, position-offset), then eval the ae_L62-trained AR
(anchor) on each cell using fl_big's own responses as <text>. The (layer, offset) where
anchor's FVE PEAKS = the convention ae was collected with. If anchor peaks at, say,
(layer 60, off 0) at ~24%, then ae's nominal "L62" is really layers[60] output => a
2-layer capture offset explains the whole symmetric non-transfer.

posoff: 0 = last ctx token (fl_big's convention), +1 = first response token, etc.
"""
import glob, json as _json
from pathlib import Path
import numpy as np, torch
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from nla.config import load_nla_config
from nla.schema import resolve_target_scale, compute_predict_mean_baselines
from nla.utils.critic import critic_predict  # noqa
from nla.train_sft import init_critic_from_base, heldout_fve_mse
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file as _load_file

dev = "cuda"; BASE = "Qwen/Qwen3.6-27B"; N = 500
LAYERS = [59, 60, 61, 62, 63]
POSOFF = [0, 1, 2]                       # 0=last ctx token, +k = k tokens into the response
ANCHOR = sorted(glob.glob("/workspace/cnla/skip-lens/ckpts/ar_abl_anchor/iter_*"))[-1]
SIDE = "/workspace/cnla/data/ar_L62_big_trunc/train.parquet"

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True); tok.pad_token = tok.eos_token
cfg = load_nla_config(SIDE, tok)
mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
template = cfg.critic_prompt_template

# ---- stage 1: recompute fl_big activations on a (layer, posoff) grid ----
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to(dev).eval()
grab = {}
for Lh in LAYERS:
    model.model.layers[Lh].register_forward_hook(
        lambda m, i, o, LL=Lh: grab.__setitem__(LL, (o[0] if isinstance(o, tuple) else o).detach()))

t = pq.ParquetFile("/workspace/cnla/skip-lens/data/fl_big/sft_train.parquet").read_row_group(0)
ctx = t.column("ctx_text").to_pylist()
resp = t.column("response").to_pylist()

# grid[(L,off)] = list of activation vectors; explanations = responses (shared across cells)
grid = {(L, off): [] for L in LAYERS for off in POSOFF}
expls = []
maxoff = max(POSOFF)
n_used = 0
for i in range(len(ctx)):
    if n_used >= N:
        break
    e = (resp[i] or "").strip()
    if not e:
        continue
    cids = tok.encode(ctx[i], add_special_tokens=False)[-500:]
    rids = tok.encode(resp[i], add_special_tokens=False)[:maxoff + 1]
    if len(cids) < 2 or len(rids) < maxoff + 1:
        continue
    full = torch.tensor([cids + rids], device=dev)
    with torch.no_grad():
        model(input_ids=full)
    base = len(cids) - 1                 # index of last ctx token
    for L in LAYERS:
        h = grab[L][0]                   # [T, d]
        for off in POSOFF:
            grid[(L, off)].append(h[base + off].float().cpu().numpy())
    expls.append(e)
    n_used += 1
print(f"[grid] recomputed {n_used} rows x {len(LAYERS)} layers x {len(POSOFF)} offsets", flush=True)
del model
torch.cuda.empty_cache()

# ---- stage 2: load ae-trained anchor AR, eval FVE on every cell ----
m = _json.loads((Path(ANCHOR) / "ar_meta.json").read_text())
critic = init_critic_from_base(BASE, m["ar_num_layers"], torch.bfloat16, None,
                               device_map=None, max_memory=None,
                               strip_final_norm=m.get("final_norm_stripped", True)).to(dev).eval()
inject_adapter_in_model(LoraConfig(r=m["lora_r"], lora_alpha=m["lora_alpha"], lora_dropout=0.0,
                                   bias="none", task_type="CAUSAL_LM", use_rslora=True,
                                   target_modules=m["target_modules"]), critic.backbone)
sd = _load_file(str(Path(ANCHOR) / "ar_lora_value_head.safetensors"))
critic.load_state_dict(sd, strict=False)
for p in critic.parameters():
    p.requires_grad_(False)
critic.eval()
print(f"[anchor] {ANCHOR}", flush=True)

print("\n=== anchor (ae_L62-trained) FVE % on recomputed fl_big activations ===", flush=True)
print("     " + "  ".join(f"off+{o}" for o in POSOFF), flush=True)
best = (-9, None)
for L in LAYERS:
    cells = []
    for off in POSOFF:
        acts = grid[(L, off)]
        pairs = list(zip(expls, acts))
        v = torch.tensor(np.stack(acts)).float()
        bl = compute_predict_mean_baselines(v, mse_scale_f)[1]
        mse, ns = heldout_fve_mse(critic, tok, pairs, template, mse_scale_f, dev)
        fve = 100 * (1 - mse / bl)
        cells.append(fve)
        if fve > best[0]:
            best = (fve, (L, off))
    print(f"L{L}: " + "  ".join(f"{c:5.1f}%" for c in cells), flush=True)
print(f"\n>>> anchor peaks at layer/offset {best[1]} = {best[0]:.1f}% "
      f"(fl_big's own convention is L62/off0; anchor's peak reveals ae's convention)", flush=True)
