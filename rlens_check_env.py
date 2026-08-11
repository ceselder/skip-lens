"""Env sanity checks for R-Lens on the B300 box (gpu4).

1. Does fla FusedRMSNormGated(x, g) numerically match Qwen3_5RMSNormGated?
2. What does ALL_ATTENTION_FUNCTIONS.get_interface("eager", default) return?
3. Remaining text_config fields (linear head dims, vocab, n_layers).
4. Does torch_chunk_gated_delta_rule match fla chunk_gated_delta_rule (fwd)?
"""
import torch
import torch.nn.functional as F

from transformers.models.qwen3_5 import modeling_qwen3_5 as m

torch.manual_seed(0)
dev = "cuda"

# --- 1. gated norm equivalence ---
hd = 128
fla_norm = m.FusedRMSNormGated(hd, eps=1e-6, activation="silu").to(dev, torch.float32)
qwen_norm = m.Qwen3_5RMSNormGated(hd, eps=1e-6).to(dev, torch.float32)
with torch.no_grad():
    w = torch.randn(hd, device=dev).abs() + 0.5
    fla_norm.weight.copy_(w)
    qwen_norm.weight.copy_(w)
    x = torch.randn(64, hd, device=dev, dtype=torch.bfloat16)
    g = torch.randn(64, hd, device=dev, dtype=torch.bfloat16)
    out_fla = fla_norm(x.float(), g.float())
    out_qwen = qwen_norm(x.float(), g.float())
    print("gated-norm max abs diff (fla vs qwen fallback):",
          (out_fla - out_qwen).abs().max().item())
    # my detached reimplementation (forward must match exactly)
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    mine = (xf * rstd) * w * F.silu(g.float())
    print("gated-norm max abs diff (mine vs fla):", (mine - out_fla).abs().max().item())
    print("gated-norm max abs diff (mine vs qwen):", (mine - out_qwen).abs().max().item())

# --- 2. attention interface dispatch ---
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
try:
    iface = ALL_ATTENTION_FUNCTIONS.get_interface("eager", m.eager_attention_forward)
    print("get_interface('eager', default) ->", iface)
    print("  is module-global eager_attention_forward:", iface is m.eager_attention_forward)
except Exception as e:
    print("get_interface('eager') raised:", repr(e))
try:
    iface2 = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", m.eager_attention_forward)
    print("get_interface('sdpa', default) ->", iface2)
except Exception as e:
    print("get_interface('sdpa') raised:", repr(e))

# --- 3. config fields ---
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B")
tc = cfg.get_text_config()
for k in ["num_hidden_layers", "hidden_size", "vocab_size", "num_attention_heads",
          "num_key_value_heads", "head_dim", "linear_num_value_heads",
          "linear_num_key_heads", "linear_key_head_dim", "linear_value_head_dim",
          "linear_conv_kernel_dim", "rms_norm_eps", "hidden_act"]:
    print(f"  {k} = {getattr(tc, k, 'MISSING')}")
lt = tc.layer_types
print("  layer_types[40:64] =", lt[40:64])
print("  n full_attention in 42..62:", [i for i in range(42, 63) if lt[i] == "full_attention"])

# --- 4. chunk rule: torch fallback vs fla kernel forward ---
B, T, HK, HV, DK, DV = 2, 96, 8, 16, 64, 64
q = torch.randn(B, T, HK, DK, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, T, HK, DK, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, T, HV, DV, device=dev, dtype=torch.bfloat16)
g = -torch.rand(B, T, HV, device=dev, dtype=torch.float32) * 2
beta = torch.rand(B, T, HV, device=dev, dtype=torch.bfloat16)
qr = q.repeat_interleave(HV // HK, dim=2)
kr = k.repeat_interleave(HV // HK, dim=2)
out_torch, _ = m.torch_chunk_gated_delta_rule(
    qr, kr, v, g, beta, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=True)
try:
    out_fla, _ = m.chunk_gated_delta_rule(
        qr, kr, v, g=g, beta=beta, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True)
    d = (out_torch.float() - out_fla.float()).abs()
    print("chunk rule fwd max/mean abs diff (torch fallback vs fla):",
          d.max().item(), d.mean().item())
    print("  out scale (mean abs):", out_fla.float().abs().mean().item())
except Exception as e:
    print("fla chunk kernel raised:", repr(e))

print("DONE")
