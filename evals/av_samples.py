"""Generic AV sample generator: inject a held-out activation, generate, show vs the target.
Works for pastlens (past-framed prompt/target) and futurelens (future-framed) alike, reading a
FINALIZED av parquet (prompt / activation_vector / response / ctx_text). Greedy decode."""
import argparse
import torch
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from nla.utils.hooks import register_karvonen_hook
from nla.schema import compute_canonical_neighbors
from nla.datagen.injection_tokens import find_injection_token
from nla.utils.arch_adapters import resolve_text_model

ACTOR_TEMPLATE = (
    "You are shown an internal activation vector captured from a language model "
    "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
    "taken at one position and encodes what the model is about to generate next. "
    "Output the text the model most likely produces immediately after this point.\n\n"
    "<concept>{injection_char}</concept>")

ap = argparse.ArgumentParser()
ap.add_argument("--av-ckpt", required=True)
ap.add_argument("--ctx-parquet", required=True)
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--max-new", type=int, default=16)
ap.add_argument("--tag", default="")
args = ap.parse_args()
dev = "cuda"

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


def _trunc_eos(ids):
    nz = (ids == tok.eos_token_id).nonzero()
    return ids[: int(nz[0])] if nz.numel() else ids


t = pq.read_table(args.ctx_parquet)
prompts = t.column("prompt").to_pylist()
acts = t.column("activation_vector").to_pylist()
resps = t.column("response").to_pylist()
cols = t.column_names
ctxs = t.column("ctx_text").to_pylist() if "ctx_text" in cols else [""] * len(acts)

print(f"=== {args.tag} | {args.av_ckpt} | n={min(args.n, len(acts))} (greedy, {args.max_new} tok) ===", flush=True)
for i in range(min(args.n, len(acts))):
    s = tok.apply_chat_template(prompts[i], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = torch.tensor([tok.encode(s, add_special_tokens=False)], device=dev)
    model._fl_vref[0] = torch.tensor(acts[i], dtype=torch.float32, device=dev).view(1, -1)
    g = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                       max_new_tokens=args.max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    gen = tok.decode(_trunc_eos(g[0, ids.shape[1]:]), skip_special_tokens=True)
    print(f"--- {i} ---", flush=True)
    print(f"  ctx_tail : ...{(ctxs[i] or '')[-90:]!r}", flush=True)
    print(f"  GEN      : {gen!r}", flush=True)
    print(f"  TARGET   : {(resps[i] or '')!r}", flush=True)
