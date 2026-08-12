"""Shared helpers: model surgery for Qwen3.6-27B.

NOTE: Qwen3_5RMSNorm applies a `(1 + weight)` gain (Gemma-style), not `weight`.
Getting this wrong silently corrupts the readout (rel. logit error ~0.5), so the
effective gain is exported once and reused everywhere.
"""
import json
import torch

MODEL_ID = "Qwen/Qwen3.6-27B"
SRC_LAYER = 42          # source residual stream (0-based block index)
D_MODEL = 5120
V = 248320
N_LAYERS = 64


def resolve_text_model(model):
    """Return the module owning `.layers` and `.norm`."""
    m = getattr(model, "model", model)
    if hasattr(m, "language_model"):
        m = m.language_model
    assert hasattr(m, "layers") and hasattr(m, "norm"), f"bad text model: {type(m)}"
    return m


def norm_gain(model):
    """Effective elementwise gain of the final RMSNorm, empirically validated."""
    tm = resolve_text_model(model)
    w = tm.norm.weight.detach().float()
    return 1.0 + w if _is_one_plus(tm.norm) else w


def _is_one_plus(norm_mod):
    """Probe the module numerically instead of trusting the class name."""
    dev = norm_mod.weight.device
    d = norm_mod.weight.shape[0]
    x = torch.randn(1, 4, d, device=dev, dtype=norm_mod.weight.dtype)
    with torch.no_grad():
        y = norm_mod(x).float()
    xf = x.float()
    u = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    w = norm_mod.weight.detach().float()
    e_plain = (y - u * w).abs().max().item()
    e_plus = (y - u * (1 + w)).abs().max().item()
    return e_plus < e_plain


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
