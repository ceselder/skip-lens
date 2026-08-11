"""R-Lens for Qwen3.6-27B: the J-lens fitter with an LRP-linearized backward.

R-Lens = fit the exact same estimator as the Jacobian lens (jlens.fitting.fit),
but with the model's forward monkeypatched so every nonlinear GATE is
`.detach()`ed. detach() is the identity in the forward, so activations are
unchanged (up to kernel-choice round-off, see notes below); but the backward
then transports relevance only through the linear/value paths, so the
resulting `torch.autograd.grad(h_62, h_l)` matrices ARE the LRP relevance
transports R_l (same shape/convention as J_l: [d_model, d_model], rows =
output dims, transport = R @ h).

Exact detaches (see `lrp_detach_patches`):
  * Qwen3_5RMSNorm._norm         : rsqrt(mean(x^2)+eps) factor detached
                                   (norm becomes a constant diagonal scale).
  * Qwen3_5RMSNormGated.forward  : rsqrt factor AND F.silu(gate) detached.
  * Qwen3_5MLP.forward           : act_fn(gate_proj(x)) detached -> MLP linear
                                   in x through up_proj/down_proj.
  * Qwen3_5Attention.forward     : softmax attention weights detached AND
                                   sigmoid(output gate) detached -> attention
                                   linear in the value path.
  * Qwen3_5GatedDeltaNet.forward : conv SiLU linearized as
                                   conv_out * sigmoid(conv_out).detach()
                                   (keeps q/k/v live through the linear conv);
                                   query/key/beta/g all detached at the
                                   delta-rule call (value stays live), and the
                                   delta rule is routed through the pure-torch
                                   `torch_chunk_gated_delta_rule` so autograd
                                   (not the fla Triton kernel) carries the
                                   relevance. The recurrence is then exactly
                                   linear in the value stream. The gated
                                   RMSNorm afterwards is re-implemented with
                                   the Qwen3_5RMSNormGated math (matches the
                                   fla fused kernel to ~2e-6 in fp32) with
                                   rsqrt + silu(z) detached.

Environment notes for this box (checked 2026-08-11):
  * transformers 5.14.1; fla IS installed, so at runtime the unpatched model
    uses fla's chunk_gated_delta_rule Triton kernel and fla's
    FusedRMSNormGated; only causal_conv1d is missing (torch conv path used).
    The patched forward swaps both for pure-torch equivalents (forward diff is
    bf16 round-off; measured and logged by the identity check below).
  * The model must be loaded with attn_implementation="eager" so that
    (a) create_causal_mask materializes a real additive 4D mask and
    (b) the unpatched J control also runs the eager path (parity with R).

Usage (on the GPU box):
  export HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4
  /workspace/cnla_venv/bin/python rlens_fit.py            # full spike
  N_PROMPTS=2 /workspace/cnla_venv/bin/python rlens_fit.py  # smoke (resumable)

Env knobs: N_PROMPTS (24), DIM_BATCH (8), SKIP_J (0), SKIP_R (0).
"""

import contextlib
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from transformers.models.qwen3_5 import modeling_qwen3_5 as m

# ----------------------------------------------------------------------------
# Detached-forward variants. Each computes EXACTLY the same forward values as
# the original (module-for-module; the deltanet swaps fla kernels for their
# torch fallbacks, which is bf16-level round-off), with gates detached.
# ----------------------------------------------------------------------------


def _rmsnorm_norm_detached(self, x):
    # original: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps).detach()


def _rmsnorm_gated_forward_detached(self, hidden_states, gate=None):
    # Qwen3_5RMSNormGated.forward with rsqrt + silu(gate) detached.
    # LN-rule (detach rsqrt) + Identity-rule on silu(z) + Half-rule on the product.
    input_dtype = hidden_states.dtype
    x32 = hidden_states.to(torch.float32)
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.variance_epsilon).detach()
    normed = self.weight * x32.to(input_dtype)          # live (degree-1 in the recurrence readout)
    z32 = gate.to(torch.float32)
    silu_z = (z32 * torch.sigmoid(z32).detach()).to(input_dtype)   # Identity-rule: z live, sigmoid(z) const
    p = normed * silu_z                                 # both live -> degree-2 bilinear
    p = 0.5 * p + 0.5 * p.detach()                      # Half-rule
    return p.to(input_dtype)


def _mlp_forward_detached(self, x):
    # RelP: Identity-rule on SiLU + Half-rule on the gate*up product (both branches
    # are unbounded CONTENT, so relevance splits 50/50 -> gate_proj AND up_proj both live).
    # silu(u) = u*sigmoid(u): detach sigmoid(u) (Identity-rule) so u stays live.
    u = self.gate_proj(x)
    a = u * torch.sigmoid(u).detach()          # == silu(u) in value; backward carries [sigmoid(u)]_const
    p = a * self.up_proj(x)                     # both factors live -> degree-2 bilinear in x
    p = 0.5 * p + 0.5 * p.detach()             # Half-rule: restores Euler identity / conservation
    return self.down_proj(p)


def _attn_forward_detached(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
    """Qwen3_5Attention.forward, eager attention inlined, with the softmax
    attention weights and the sigmoid output gate detached."""
    if past_key_values is not None:
        raise NotImplementedError("R-lens patched attention is fit-only (no cache)")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )
    gate = gate.reshape(*input_shape, -1)

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = m.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    key_states = m.repeat_kv(key_states, self.num_key_value_groups)
    value_states = m.repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    else:
        # Fallback causal mask (eager-mode load should always provide one).
        q_len, k_len = attn_weights.shape[-2], attn_weights.shape[-1]
        neg = torch.finfo(attn_weights.dtype).min
        causal = torch.triu(
            torch.full((q_len, k_len), neg, device=attn_weights.device,
                       dtype=attn_weights.dtype),
            diagonal=1,
        )
        attn_weights = attn_weights + causal

    # LRP: attention weights are a gate -> constant in the backward.
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    ).detach()
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    # LRP: output gate detached.
    attn_output = attn_output * torch.sigmoid(gate).detach()

    attn_output = self.o_proj(attn_output)
    return attn_output, None


def _gdn_forward_detached(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
    """Qwen3_5GatedDeltaNet.forward, no-cache path only, LRP-linearized.

    Live gradient path: hidden_states -> in_proj_qkv -> conv1d (linear) ->
    x*sigmoid(x).detach() -> value stream through torch_chunk_gated_delta_rule
    (q/k/beta/g detached) -> gated RMSNorm (rsqrt + silu(z) detached) ->
    out_proj.
    """
    if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
        raise NotImplementedError("R-lens patched deltanet is fit-only (no cache)")

    hidden_states = m.apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    mixed_qkv = self.in_proj_qkv(hidden_states)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    z = self.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if cache_params is not None:
        new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
        cache_params.update_conv_state(new_conv_state, self.layer_idx)

    # original torch path: mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
    # LRP: silu(x) = x * sigmoid(x); detach the sigmoid factor, keep x live.
    conv_out = self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]]
    mixed_qkv = conv_out * torch.sigmoid(conv_out).detach()

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
    )
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    # LRP: beta / decay gates detached (constants in the backward).
    beta = b.sigmoid().detach()
    g = (-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)).detach()
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    # LRP: q/k detached; value stays live -> the whole chunked delta-rule
    # recurrence is exactly linear in the value stream. Routed through the
    # pure-torch fallback (NOT the fla Triton kernel) so plain autograd
    # carries the relevance.
    core_attn_out, _ = m.torch_chunk_gated_delta_rule(
        query.detach(),
        key.detach(),
        value,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    # Gated RMSNorm (self.norm is fla's FusedRMSNormGated at runtime here;
    # this is the Qwen3_5RMSNormGated math, which matches it to ~2e-6 fp32),
    # with the rsqrt factor and the silu(z) gate detached.
    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    eps = getattr(self.norm, "variance_epsilon", getattr(self.norm, "eps", 1e-6))
    input_dtype = core_attn_out.dtype
    x32 = core_attn_out.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps).detach()
    normed = self.norm.weight * x32.to(input_dtype)          # live
    z32 = z.to(torch.float32)
    silu_z = (z32 * torch.sigmoid(z32).detach()).to(input_dtype)   # Identity-rule: z (in_proj_z) live
    p = normed * silu_z                                      # both live -> degree-2
    p = 0.5 * p + 0.5 * p.detach()                           # Half-rule
    core_attn_out = p.to(input_dtype)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    return self.out_proj(core_attn_out)


@contextlib.contextmanager
def lrp_detach_patches():
    """Monkeypatch the Qwen3_5 forwards to the LRP-detached variants; restore
    the originals on exit. Applies at class level, so it affects every
    Qwen3_5 model in this process while active."""
    originals = {
        "rms_norm": m.Qwen3_5RMSNorm._norm,
        "rms_norm_gated": m.Qwen3_5RMSNormGated.forward,
        "mlp": m.Qwen3_5MLP.forward,
        "attn": m.Qwen3_5Attention.forward,
        "gdn": m.Qwen3_5GatedDeltaNet.forward,
    }
    m.Qwen3_5RMSNorm._norm = _rmsnorm_norm_detached
    m.Qwen3_5RMSNormGated.forward = _rmsnorm_gated_forward_detached
    m.Qwen3_5MLP.forward = _mlp_forward_detached
    m.Qwen3_5Attention.forward = _attn_forward_detached
    m.Qwen3_5GatedDeltaNet.forward = _gdn_forward_detached
    try:
        yield
    finally:
        m.Qwen3_5RMSNorm._norm = originals["rms_norm"]
        m.Qwen3_5RMSNormGated.forward = originals["rms_norm_gated"]
        m.Qwen3_5MLP.forward = originals["mlp"]
        m.Qwen3_5Attention.forward = originals["attn"]
        m.Qwen3_5GatedDeltaNet.forward = originals["gdn"]


# ----------------------------------------------------------------------------
# Spike driver
# ----------------------------------------------------------------------------

BASE = "Qwen/Qwen3.6-27B"
SRC = [int(x) for x in os.environ.get("SRC", "42,55").split(",")]
TGT = int(os.environ.get("TGT", "62"))
OUT = os.environ.get("OUT", "/workspace/cnla/results/rlens_spike")
PARQUET = "/workspace/cnla/skip-lens/data/fl_big/sft_train.parquet"


def load_prompts(n):
    import pandas as pd

    df = pd.read_parquet(PARQUET, columns=["ctx_text"])
    texts = df["ctx_text"].drop_duplicates()
    # ctx_text tops out at ~377 chars (~90 tokens); take the first n that are
    # >= 300 chars so each prompt has plenty of valid positions past skip_first.
    prompts = [t for t in texts if isinstance(t, str) and len(t) >= 300][: n]
    if len(prompts) < n:
        raise ValueError(f"only found {len(prompts)} usable prompts")
    return prompts


def identity_check(model, prompt):
    """Patched forward must equal unpatched forward (detach is fwd-identity);
    residual diff comes only from torch-vs-fla kernel round-off."""
    from jlens.hooks import ActivationRecorder

    input_ids = model.encode(prompt, max_length=128)
    acts = {}
    for tag, ctx in (("plain", contextlib.nullcontext()), ("patched", lrp_detach_patches())):
        with ctx, torch.no_grad(), ActivationRecorder(model.layers, at=[42, 55, 62]) as rec:
            model.forward(input_ids)
            acts[tag] = {l: rec.activations[l].detach().float().cpu() for l in (42, 55, 62)}
    for l in (42, 55, 62):
        d = (acts["plain"][l] - acts["patched"][l]).abs()
        scale = acts["plain"][l].abs().mean()
        print(
            f"[identity] L{l}: max|plain-patched|={d.max():.4f} "
            f"mean|.|={d.mean():.5f} vs act mean|.|={scale:.3f} "
            f"(rel mean {d.mean() / scale:.2e})",
            flush=True,
        )


def mask_check(model):
    """Assert the full-attention layers receive a real additive causal mask in
    eager mode (guards the detached-eager attention path)."""
    seen = {}

    def hook(module, args, kwargs):
        seen["mask"] = kwargs.get("attention_mask", None)

    h = model.layers[43].register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model.forward(model.encode("The quick brown fox jumps over the lazy dog. " * 4, max_length=32))
    finally:
        h.remove()
    mask = seen.get("mask")
    assert mask is not None and mask.ndim == 4, f"full-attn mask is {type(mask)}; need eager 4D mask"
    off_diag = mask[0, 0, 0, -1].item()
    assert off_diag < -1e30 or torch.isinf(torch.tensor(off_diag)), f"mask[0,0,0,-1]={off_diag}"
    print(f"[mask] full-attn mask OK: shape={tuple(mask.shape)} dtype={mask.dtype} "
          f"masked value={off_diag:.3e}", flush=True)


def run_fit(model, prompts, tag, dim_batch, patched):
    from jlens.fitting import fit

    ckpt = f"{OUT}/fit_ckpt_{tag}.pt"
    t0 = time.time()
    ctx = lrp_detach_patches() if patched else contextlib.nullcontext()
    with ctx:
        lens = fit(
            model,
            prompts,
            source_layers=SRC,
            target_layer=TGT,
            dim_batch=dim_batch,
            max_seq_len=128,
            checkpoint_path=ckpt,
            checkpoint_every=1,
            resume=True,
        )
    print(f"[{tag}] fit done in {time.time() - t0:.0f}s over {lens.n_prompts} prompts", flush=True)
    for l in SRC:
        M = lens.jacobians[l].float()
        np.save(f"{OUT}/{tag}_L{l}_to_L{TGT}.npy", M.numpy().astype("float32"))
        print(
            f"  [{tag}] L{l}: ||M||_F/sqrt(d)={M.norm().item() / np.sqrt(model.d_model):.4f} "
            f"finite={bool(torch.isfinite(M).all())}",
            flush=True,
        )
    lens.save(f"{OUT}/lens_{tag}.pt")
    return lens


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    n_prompts = int(os.environ.get("N_PROMPTS", "24"))
    dim_batch = int(os.environ.get("DIM_BATCH", "8"))
    skip_j = os.environ.get("SKIP_J", "0") == "1"
    skip_r = os.environ.get("SKIP_R", "0") == "1"
    os.makedirs(OUT, exist_ok=True)

    prompts = load_prompts(n_prompts)
    with open(f"{OUT}/prompts.json", "w") as f:
        json.dump(prompts, f)
    print(f"[main] {len(prompts)} prompts | src={SRC} tgt=L{TGT} dim_batch={dim_batch}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hf import from_hf

    tok = AutoTokenizer.from_pretrained(BASE)
    hf = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()
    model = from_hf(hf, tok)
    print(f"[main] LensModel: n_layers={model.n_layers} d_model={model.d_model} "
          f"attn_impl={hf.config._attn_implementation}", flush=True)

    mask_check(model)
    identity_check(model, prompts[0])

    if not skip_j:
        run_fit(model, prompts, "J", dim_batch, patched=False)
    if not skip_r:
        run_fit(model, prompts, "R", dim_batch, patched=True)
    print("=== RLENS SPIKE FIT DONE ===", flush=True)


if __name__ == "__main__":
    main()
