"""2x2 AR transfer matrix: does the failure to transfer go BOTH ways?

rows = AR checkpoint (anchor = trained on ae_L62 50,925 = the 23.3% one;
       big_trunc = trained on fl_big 485k), cols = eval dataset.
Symmetric low off-diagonal => the two collections are genuinely different
distributions (each AR only fits its own) => collection convention changed.
"""
import glob, json as _json
from pathlib import Path
import numpy as np, torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from nla.config import load_nla_config
from nla.schema import resolve_target_scale, compute_predict_mean_baselines
from nla.utils.critic import critic_predict  # noqa
from nla.train_sft import init_critic_from_base, heldout_fve_mse
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import load_file as _load_file

dev = "cuda"; BASE = "Qwen/Qwen3.6-27B"; N = 800


def latest(base):
    ds = sorted(glob.glob(base + "/iter_*"))
    return ds[-1] if ds else None


CKPTS = {
    "anchor (ae_L62-trained, ~23.3%)": latest("/workspace/cnla/skip-lens/ckpts/ar_abl_anchor"),
    "big_trunc (fl_big-trained)": "/workspace/cnla/skip-lens/ckpts/ar_L62_big_trunc/iter_0006000",
}
SETS = {
    "ae_heldout": "/workspace/cnla/data/ar_ablation/heldout.parquet",
    "fl_big": "/workspace/cnla/data/ar_L62_big_trunc/train.parquet",
}
SIDE = "/workspace/cnla/data/ar_L62_big_trunc/train.parquet"
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True); tok.pad_token = tok.eos_token
cfg = load_nla_config(SIDE, tok)
mse_scale_f = resolve_target_scale(cfg.mse_scale, cfg.d_model)
template = cfg.critic_prompt_template
print(f"[cfg] mse_scale_f={mse_scale_f:.3f} template={template!r}", flush=True)
print(f"[ckpts] anchor={CKPTS['anchor (ae_L62-trained, ~23.3%)']}", flush=True)


def pairs_from_prompt(parquet, n):
    t = pq.ParquetFile(parquet).read_row_group(0)
    pr = t.column("prompt").to_pylist()[:n]
    ac = t.column("activation_vector").combine_chunks()
    acts = ac.flatten().to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(ac), -1)[:n]
    return [(s.split("<text>", 1)[-1].rsplit("</text>", 1)[0], a) for s, a in zip(pr, acts)]


PAIRS = {k: pairs_from_prompt(v, N) for k, v in SETS.items()}
BL = {k: compute_predict_mean_baselines(torch.tensor(np.stack([a for _, a in p])).float(), mse_scale_f)[1]
      for k, p in PAIRS.items()}
print("[baselines] " + " ".join(f"{k}={BL[k]:.4f}" for k in SETS), flush=True)


def load_critic(ck):
    m = _json.loads((Path(ck) / "ar_meta.json").read_text())
    c = init_critic_from_base(BASE, m["ar_num_layers"], torch.bfloat16, None,
                              device_map=None, max_memory=None,
                              strip_final_norm=m.get("final_norm_stripped", True)).to(dev).eval()
    inject_adapter_in_model(LoraConfig(r=m["lora_r"], lora_alpha=m["lora_alpha"], lora_dropout=0.0,
                                       bias="none", task_type="CAUSAL_LM", use_rslora=True,
                                       target_modules=m["target_modules"]), c.backbone)
    sd = _load_file(str(Path(ck) / "ar_lora_value_head.safetensors"))
    c.load_state_dict(sd, strict=False)
    for p in c.parameters():
        p.requires_grad_(False)
    return c.eval()


print("\n=== TRANSFER MATRIX (FVE %) — rows=AR ckpt, cols=eval set ===", flush=True)
hdr = " " * 34 + " | " + " | ".join(f"{sk:>12s}" for sk in SETS)
print(hdr, flush=True)
for name, ck in CKPTS.items():
    if not ck:
        print(f"{name:34s} | MISSING"); continue
    c = load_critic(ck)
    cells = []
    for sk, pairs in PAIRS.items():
        mse, ns = heldout_fve_mse(c, tok, pairs, template, mse_scale_f, dev)
        cells.append(f"{100*(1-mse/BL[sk]):11.1f}%")
    print(f"{name:34s} | " + " | ".join(cells), flush=True)
    del c
    torch.cuda.empty_cache()
