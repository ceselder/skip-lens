"""Push the normalized-mean-centering skip-lens (the +0.119 @ L42 winner) to a private HF repo,
bundled with the mean-direction vectors it needs + a model card that spells out the wiring."""
import os
import shutil
from huggingface_hub import HfApi

CK = "/workspace/cnla/skip-lens/ckpts/skiplens_meansub_norm_500/iter_0000300"
MEANS = "/workspace/cnla/skip-lens/data/meansub"
STAGE = "/workspace/cnla/skip-lens/hf_stage_meansub_norm"
REPO = "ceselder/skip-lens-qwen36-27b-meansub-norm"
TOKEN = os.environ["HF_TOKEN"]

os.makedirs(STAGE, exist_ok=True)
for f in ["adapter_config.json", "adapter_model.safetensors", "nla_meta.yaml",
          "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]:
    shutil.copy(f"{CK}/{f}", f"{STAGE}/{f}")
shutil.copy(f"{MEANS}/mean_dir_L62_train.npy", f"{STAGE}/mean_dir_L62_train.npy")
shutil.copy(f"{MEANS}/mean_test_dir_by_layer.npz", f"{STAGE}/mean_test_dir_by_layer.npz")

README = r"""---
base_model: Qwen/Qwen3.6-27B
library_name: peft
tags: [interpretability, futurelens, skip-lens, activation-oracle, mean-centering]
---

# skip-lens (Qwen3.6-27B) — normalized mean-centering variant

A **futurelens / skip-lens**: a LoRA adapter on `Qwen/Qwen3.6-27B` that reads a residual-stream
activation you *inject* and verbalizes the model's internal "workspace" (what it is about to compute).
Trained to inject the **penultimate (L62)** activation; at test you feed an **intermediate** layer
(e.g. **L42**) to surface the answer-distinct workspace.

**Why this variant:** injecting the *normalized deviation from the mean direction* instead of the raw
activation lifts fed-L42 workspace agreement from **0.41 → 0.53** (+0.12, ~6 SEM), nearly matching a
300x-larger 150k-pair lens. Plain magnitude-mean-subtraction (+0.01) and ZCA whitening (+0.02) do NOT
help — **mean-centering in unit space is the trick.** (500-pair proof-of-concept.)

## How it works (the important part — the adapter alone is not enough)

The source activation must be **mean-centered in unit space** before injection:

    v = h/||h||  -  mean_dir_layer          # unit vector minus the layer's mean DIRECTION

then injected **norm-matched** (Karvonen et al. 2025) at the output of **decoder block 1**, at the
single injection token `㈜` (id 158983), whose residual `h_p` is overwritten:

    h_p  <-  h_p + ||h_p|| * v / ||v||       # magnitude of v is discarded; only its direction matters

The prompt is the actor template (see `nla_meta.yaml -> prompt_templates.actor`) with `㈜` in the
`<concept>...</concept>` slot, applied via the chat template with `enable_thinking=False`.

### mean-direction files (included)
- `mean_dir_L62_train.npy`  — (5120,) mean of UNIT L62 activations over the fl_big train distribution
  (subtract this from the L62 unit activation; used during training).
- `mean_test_dir_by_layer.npz` — per fed layer `{"62","55","48","42","34","26"}`, each (5120,) the mean
  of UNIT activations at that layer over the eval distribution. **At test, subtract the fed layer's
  entry** (e.g. `z["42"]` when feeding L42).

## Minimal wiring (feed L42)

```python
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
# repo helpers: register_karvonen_hook (block-1 norm-matched inject), ACTOR_TEMPLATE, _prompt_ids
from nla.utils.hooks import register_karvonen_hook
from nla.schema import compute_canonical_neighbors
from nla.datagen.injection_tokens import find_injection_token
from fl_common import ACTOR_TEMPLATE, _prompt_ids

dev = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-27B", torch_dtype=torch.bfloat16).to(dev).eval()
model = PeftModel.from_pretrained(base, "ceselder/skip-lens-qwen36-27b-meansub-norm").eval()
inj, inj_id = find_injection_token(tok); L, R = compute_canonical_neighbors(tok, ACTOR_TEMPLATE, inj, inj_id)
vref = [None]; register_karvonen_hook(model, vref, inj_id, L, R); model._fl_vref = vref

mean_dir = torch.tensor(np.load("mean_test_dir_by_layer.npz")["42"]).float().to(dev)  # L42 mean direction

# h42 = the L42 residual you grabbed from the base model at your position of interest (shape [5120])
h = h42.float()
v = h / (h.norm() + 1e-8) - mean_dir                 # <-- mean-centering in unit space (the trick)
pt = _prompt_ids(tok, dev)
model._fl_vref[0] = v.view(1, -1)                     # injection is norm-matched, so scale of v is irrelevant
out = model.generate(pt, max_new_tokens=24, do_sample=True, temperature=0.7, top_p=0.95)
print(tok.decode(out[0, pt.shape[1]:], skip_special_tokens=True))
model._fl_vref[0] = None
```

The reference eval is the repo script `evals/fedlayer_meansub_eval.py --normalize-first
--sub-mean-npz mean_test_dir_by_layer.npz`.

## Caveats
- **500-pair proof-of-concept**, LoRA r64 α16 rsLoRA scope=all, AV-SFT.
- **Trained on RAW-text activations.** Qwen3.6-27B is a chat model; on raw text it runs off-distribution.
  For a chat-native pipeline, harvest the source activation chat-natively (user turn +
  `add_generation_prompt=True, enable_thinking=False`, at the assistant anchor) — and ideally retrain
  the lens on chat-native + mean-centered activations. The mean-centering *recipe* carries over.
- Full report (mean-centering vs whitening vs raw, geometry, chat-native benchmark):
  http://5.78.192.0/reports/view/skiplens-layer-ablation/report.html
"""
open(f"{STAGE}/README.md", "w").write(README)

api = HfApi(token=TOKEN)
api.create_repo(REPO, private=True, exist_ok=True, repo_type="model")
api.upload_folder(folder_path=STAGE, repo_id=REPO, repo_type="model")
print("UPLOAD_DONE", REPO)
