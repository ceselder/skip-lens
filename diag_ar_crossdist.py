"""Diagnose the fl_big-trained AR's 23%-train / 7%-heldout gap.

Reload the fl_big-trained critic and measure reconstruction FVE three ways:
  A  fl_big own data (explanation from its prompt <text>) -> expect ~train (~23%) if AR is fine
  B  50,925 heldout via `response` extraction (EXACTLY what the training run's heldout used) -> the 7%
  C  50,925 heldout via its prompt <text> (same format path as A)
If A>>B: genuine cross-distribution. If C>>B: the response-extraction path was the artifact.
"""
import json as _json
from pathlib import Path
import numpy as np, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from nla.config import load_nla_config
from nla.schema import resolve_target_scale, compute_predict_mean_baselines
from nla.utils.critic import critic_predict  # noqa
from nla.train_sft import init_critic_from_base, heldout_fve_mse, load_heldout_explanation_pairs
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file as _load_file

dev = "cuda"
BASE = "Qwen/Qwen3.6-27B"
AR = Path("/workspace/cnla/skip-lens/ckpts/ar_L62_big_trunc/iter_0006000")
SIDE = "/workspace/cnla/data/ar_L62_big_trunc/train.parquet"
FL = "/workspace/cnla/data/ar_L62_big_trunc/train.parquet"
HO = "/workspace/cnla/data/ar_ablation/heldout.parquet"
N = 1000

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tok.pad_token = tok.eos_token
cfg = load_nla_config(SIDE, tok)
mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
template = cfg.critic_prompt_template
print(f"template={template!r} mse_scale_f={mse_scale_f}", flush=True)

ar_meta = _json.loads((AR / "ar_meta.json").read_text())
critic = init_critic_from_base(
    BASE, ar_meta["ar_num_layers"], torch.bfloat16, None,
    device_map=None, max_memory=None,
    strip_final_norm=ar_meta.get("final_norm_stripped", True)).to(dev).eval()
inject_adapter_in_model(LoraConfig(
    r=ar_meta["lora_r"], lora_alpha=ar_meta["lora_alpha"], lora_dropout=0.0,
    bias="none", task_type="CAUSAL_LM", use_rslora=True,
    target_modules=ar_meta["target_modules"]), critic.backbone)
sd = _load_file(str(AR / "ar_lora_value_head.safetensors"))
miss, unexp = critic.load_state_dict(sd, strict=False)
assert not unexp and sum(1 for k in sd if "lora_" in k) > 0, f"load mismatch unexp={unexp[:3]}"
for p in critic.parameters():
    p.requires_grad_(False)
critic.eval()
print(f"[loaded] {len(sd)} tensors, ar_num_layers={ar_meta['ar_num_layers']}", flush=True)


def pairs_from_prompt(parquet, n):
    t = pq.ParquetFile(parquet).read_row_group(0)
    pr = t.column("prompt").to_pylist()[:n]
    ac = t.column("activation_vector").combine_chunks()
    acts = ac.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(ac), -1)[:n]
    out = []
    for s, a in zip(pr, acts):
        ex = s.split("<text>", 1)[-1].rsplit("</text>", 1)[0]
        out.append((ex, a))
    return out


def baseline_of(pairs):
    v = torch.tensor(np.stack([a for _, a in pairs]), dtype=torch.float32)
    _, raw = compute_predict_mean_baselines(v, mse_scale_f)
    return raw


def run(tag, pairs):
    bl = baseline_of(pairs)
    mse, ns = heldout_fve_mse(critic, tok, pairs, template, mse_scale_f, dev)
    exlen = np.mean([len(e) for e, _ in pairs])
    print(f"{tag}: n={ns} mean<expl>chars={exlen:.1f} mse={mse:.4f} baseline={bl:.4f} "
          f"FVE={100*(1-mse/bl):.1f}%", flush=True)


run("A_flbig_own (prompt<text>)", pairs_from_prompt(FL, N))
run("B_50925_heldout (response = what-run-used)", load_heldout_explanation_pairs(HO, N))
run("C_50925_heldout (prompt<text>)", pairs_from_prompt(HO, N))
