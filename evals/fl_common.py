"""Shared helpers for future-lens evals: load the AV, inject an activation
(norm-matched, Karvonen-style, at block 1), sample a rollout; plus J-lens /
logit-lens top-k readouts.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from nla.utils.hooks import register_karvonen_hook
from nla.schema import compute_canonical_neighbors
from nla.datagen.injection_tokens import find_injection_token
from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_model


def base_causal(model):
    """Unwrap a PeftModel to the underlying CausalLM (has .model + lm_head)."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return resolve_text_model(inner)


def decoder_layers(model):
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return resolve_decoder_layers(inner)

ACTOR_TEMPLATE = (
    "You are shown an internal activation vector captured from a language model "
    "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
    "taken at one position and encodes what the model is about to generate next. "
    "Output the text the model most likely produces immediately after this point.\n\n"
    "<concept>{injection_char}</concept>"
)


def load_av(base_ckpt, adapter_dir=None, affine_path=None, device="cuda", dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(base_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        base_ckpt, torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()
    inj_char, inj_id = find_injection_token(tok)
    left, right = compute_canonical_neighbors(tok, ACTOR_TEMPLATE, inj_char, inj_id)
    vref = [None]
    if affine_path:
        # affine-only variant: frozen base + learned affine at the inject layer
        ck = torch.load(affine_path, map_location=device)
        Aw = ck["weight"].to(device).float(); Ab = ck["bias"].to(device).float()
        inj_layer = ck.get("inject_layer", 1)
        cap = {"ids": None}
        model.get_input_embeddings().register_forward_hook(
            lambda m, i, o: cap.__setitem__("ids", i[0] if i else None) or o)
        def affine_hook(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            v = vref[0]
            if v is None or cap["ids"] is None:
                return out
            ids = cap["ids"]
            for b in range(ids.shape[0]):
                pos = (ids[b] == inj_id).nonzero()
                if pos.numel():
                    p = int(pos[0]); add = Aw @ v[b].to(torch.float32) + Ab
                    h = t[b, p]; t[b, p] = h + h.norm() * add.to(t.dtype) / (add.norm() + 1e-6)
            return (t, *out[1:]) if isinstance(out, tuple) else t
        decoder_layers(model)[inj_layer].register_forward_hook(affine_hook)
    else:
        if adapter_dir:
            model = PeftModel.from_pretrained(model, adapter_dir).eval()
        register_karvonen_hook(model, vref, inj_id, left, right)
    model._fl_vref = vref
    model._fl_tok = tok
    model._fl_inj = (inj_id, left, right)
    model._fl_char = inj_char
    return model, tok


def _prompt_ids(tok, device):
    inj_char, _ = find_injection_token(tok)
    content = ACTOR_TEMPLATE.format(injection_char=inj_char)
    s = tok.apply_chat_template([{"role": "user", "content": content}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(s, add_special_tokens=False)
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def av_rollout(model, tok, activation, device="cuda", max_new_tokens=24,
               n_samples=1, temperature=0.8):
    """activation: 1D tensor/np [d]. Returns list of decoded continuations."""
    act = torch.as_tensor(activation, dtype=torch.float32, device=device).view(1, -1)
    pt = _prompt_ids(tok, device)
    outs = []
    for k in range(n_samples):
        model._fl_vref[0] = act
        try:
            gen = model.generate(
                input_ids=pt, attention_mask=torch.ones_like(pt),
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0), temperature=max(temperature, 1e-5),
                top_p=0.95, pad_token_id=tok.eos_token_id)
        finally:
            model._fl_vref[0] = None
        outs.append(tok.decode(gen[0, pt.shape[1]:], skip_special_tokens=True).strip())
    return outs


@torch.no_grad()
def lens_topk(model, vec, k=10):
    """Apply the model's final norm + unembed to `vec` (a d-dim residual) and
    return the top-k tokens. Use for both logit-lens (vec=h) and J-lens
    (vec = J_l h) readouts."""
    bc = base_causal(model)
    norm = bc.model.norm
    lm_head = bc.get_output_embeddings()
    w = lm_head.weight
    v = torch.as_tensor(vec, dtype=w.dtype, device=w.device).view(1, -1)
    hn = norm(v)
    logits = lm_head(hn).float().view(-1)
    top = torch.topk(logits, k)
    return top.indices.tolist(), top.values.tolist()
