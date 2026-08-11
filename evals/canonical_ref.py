"""Single canonical 'what the model would say' reference: base model (NO LoRA), greedy-decode the
first k tokens from each disagreement-item's context, mean per-token logprob. One number = the
logprob ceiling any faithful readout could hit. Writes data/canonical_ref.json."""
import glob, json, os
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3.6-27B"; dev = "cuda"; K = 3; CTX = 256
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                             attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)

prompts = []
for f in sorted(glob.glob("/workspace/cnla/skip-lens/evals/datasets_fed/lens-eval-*.json")):
    for it in json.load(open(f))["items"]:
        prompts.append(it["prompt"])

vals = []
for n, p in enumerate(prompts):
    cur = tok(p, return_tensors="pt").input_ids[:, -CTX:].to(dev)
    lp = 0.0
    for _ in range(K):
        logits = model(cur).logits[0, -1].float()
        t = int(logits.argmax())
        lp += torch.log_softmax(logits, -1)[t].item()
        cur = torch.cat([cur, torch.tensor([[t]], device=dev)], 1)
    vals.append(lp / K)
    if n % 100 == 0:
        print(f"[ref] {n}/{len(prompts)}", flush=True)

ref = float(np.mean(vals))
json.dump({"canonical_greedy_ref": ref, "k": K, "n": len(vals)},
          open("/workspace/cnla/results/layerabl/canonical_ref.json", "w"), indent=1)
print(f"CANONICAL_REF {ref:.4f} (greedy first-{K}-token mean logprob, base model, n={len(vals)})", flush=True)
