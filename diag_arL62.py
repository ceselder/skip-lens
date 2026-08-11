"""Direct check of the RL's AR (ar_L62/iter_0001500): converged or undertrained?
Eval its held-out FVE on the ae_L62 heldout (its own distribution)."""
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

dev = "cuda"; BASE = "Qwen/Qwen3.6-27B"
AR = "/workspace/cnla/ckpts/ar_L62/iter_0001500"
SIDE = "/workspace/cnla/data/ar_L62_big_trunc/train.parquet"      # for mse_scale + template
HO = "/workspace/cnla/data/ar_ablation/heldout.parquet"          # ae_L62 in-dist heldout
N = 800

tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True); tok.pad_token = tok.eos_token
cfg = load_nla_config(SIDE, tok)
mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
template = cfg.critic_prompt_template
m = _json.loads((Path(AR) / "ar_meta.json").read_text())
critic = init_critic_from_base(BASE, m["ar_num_layers"], torch.bfloat16, None, device_map=None,
                               max_memory=None, strip_final_norm=m.get("final_norm_stripped", True)).to(dev).eval()
inject_adapter_in_model(LoraConfig(r=m["lora_r"], lora_alpha=m["lora_alpha"], lora_dropout=0.0,
                                   bias="none", task_type="CAUSAL_LM", use_rslora=True,
                                   target_modules=m["target_modules"]), critic.backbone)
sd = _load_file(str(Path(AR) / "ar_lora_value_head.safetensors"))
critic.load_state_dict(sd, strict=False)
for p in critic.parameters():
    p.requires_grad_(False)
critic.eval()

pairs = load_heldout_explanation_pairs(HO, N)
v = torch.tensor(np.stack([a for _, a in pairs])).float()
bl = compute_predict_mean_baselines(v, mse_scale_f)[1]
mse, ns = heldout_fve_mse(critic, tok, pairs, template, mse_scale_f, dev)
print(f"[ar_L62/iter_0001500] ae_L62 heldout: n={ns} mse={mse:.4f} baseline={bl:.4f} FVE={100*(1-mse/bl):.1f}%", flush=True)
print(f"  (reference: ar_abl_anchor = 23.9% on this set; ~24% = single-activation ceiling from head/data ablation)", flush=True)
print(f"  read position: template ends {template[-20:]!r} -> critic reads last token = the <summary> anchor", flush=True)
