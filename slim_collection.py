"""Slim the skip-lens collection to 4 items + rewrite the two future-lens cards accurately."""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
SLUG = "ceselder/skip-lens-multi-token-nla-lenses-qwen36-27b-6a7a17e3a421b650a62dae79"
KEEP = {
    "skip-lens-qwen36-27b-repeatafterme",
    "skip-lens-qwen36-27b-naive-futurelens",
    "skip-lens-qwen36-27b-futurelens-rl",
    "skip-lens-qwen36-27b-cnla-longhorizon",
}

col = api.get_collection(SLUG)
for it in col.items:
    short = it.item_id.split("/", 1)[-1]
    if short in KEEP:
        print("kept   ", it.item_id, flush=True)
    else:
        api.delete_collection_item(SLUG, item_object_id=it.item_object_id, missing_ok=True)
        print("removed", it.item_id, flush=True)

api.update_collection_metadata(
    SLUG,
    description="Headline skip-lens lenses on Qwen3.6-27B: repeat-after-me, future-lens, "
                "the RL future-lens, and our best compositional-NLA (long-horizon).")

FM = """---
base_model: Qwen/Qwen3.6-27B
library_name: peft
pipeline_tag: text-generation
tags:
- skip-lens
- interpretability
- nla
- future-lens
- qwen3.6-27b
license: apache-2.0
---
"""

NAIVE = FM + """
# Skip-Lens · Future-lens (the normal one)

Part of the **skip-lens** project (multi-token NL activation lenses on Qwen3.6-27B).

**This is the plain/normal future-lens.** A LoRA adapter (r=64, α=16, rsLoRA) on `Qwen/Qwen3.6-27B`
trained by **SFT** on 485,000 (layer-62 activation, future-tokens) pairs. Given a single residual
activation injected at a fixed position (Karvonen norm-matched, char `㈜` / id `158983`), it decodes
the text the model was about to produce next.

**Sibling:** `skip-lens-qwen36-27b-futurelens-rl` is the same lens after GRPO RL against a frozen
reconstruction reward. This repo is the SFT-only base future-lens.

Load as a PEFT adapter (`PeftModel.from_pretrained`) + inject a layer-62 activation via the Karvonen hook.
"""

RL = FM + """
# Skip-Lens · Future-lens after RL ("yolo" variant)

Part of the **skip-lens** project (multi-token NL activation lenses on Qwen3.6-27B).

The future-lens further trained with **GRPO RL**, where the reward is how well a **frozen activation
reconstructor (AR)** recovers the layer-62 activation from this lens's generated explanation.

> **Naming note.** This is the checkpoint we informally called the "yolo NLA." To be precise: it is
> RL'd **against a frozen AR reward model** (a separate, pre-trained AR passed as `--ar-ckpt`) — it is
> **not** jointly/co-trained with the AR. The frozen-AR reward is exactly the setup used throughout;
> no checkpoint here bundles a co-trained AR.

**Sibling:** `skip-lens-qwen36-27b-naive-futurelens` is the plain SFT future-lens (no RL).

Load as a PEFT adapter (`PeftModel.from_pretrained`) + inject a layer-62 activation via the Karvonen hook.
"""

api.upload_file(path_or_fileobj=NAIVE.encode(), path_in_repo="README.md",
                repo_id="ceselder/skip-lens-qwen36-27b-naive-futurelens", repo_type="model")
api.upload_file(path_or_fileobj=RL.encode(), path_in_repo="README.md",
                repo_id="ceselder/skip-lens-qwen36-27b-futurelens-rl", repo_type="model")
print("cards updated", flush=True)

col = api.get_collection(SLUG)
print("FINAL collection items:", [i.item_id for i in col.items], flush=True)
print("DONE", flush=True)
