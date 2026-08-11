"""Upload the matched-L42 skip-lens (source-layer ablation) to HF."""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
SRC = "/workspace/cnla/skip-lens/ckpts/skiplens_L42_150k/iter_0002344"
REPO = "ceselder/skip-lens-qwen36-27b-L42-matched"
BASE = "Qwen/Qwen3.6-27B"

CARD = f"""---
base_model: {BASE}
library_name: peft
pipeline_tag: text-generation
tags:
- skip-lens
- interpretability
- nla
- future-lens
- ablation
- qwen3.6-27b
license: apache-2.0
---

# Skip-Lens · source-layer ablation · MATCHED-L42 arm

The **matched-L42** control from the skip-lens source-layer ablation on Qwen3.6-27B. A future-lens
activation-decoder (LoRA r64 α16 rsLoRA, all modules) trained by AV-SFT to reconstruct the on-policy
future continuation from an injected residual activation — but with the **source layer = block 42**
(an intermediate), rather than the standard **L62** (penultimate).

- **Data:** 150k on-policy FineWeb continuations (`fl_big` subset), `activation_vector` re-extracted at
  block-42 last-context-token. Token-matched to the L62 (mismatch/skip-lens) arm — the *only* difference
  is the source layer.
- **Recipe:** `train_sft --mode av`, effective batch 16×4=64, 1 epoch (2,344 steps). Verbatim the
  canonical futurelens recipe.
- **Result:** fed the raw L42 residual on the disagreement set, this matched decoder **leaks the surface
  answer more** (agree_answer 0.39 vs the L62-trained arm's 0.33) and reads the workspace slightly less
  (agree_jlens 0.61 vs 0.65) — i.e. training directly on L42 drifts toward the L42→answer shortcut. The
  train/feed **mismatch** (train L62, feed L42) is what keeps the lens from decoding the output.

## Loading

PEFT LoRA adapter on `{BASE}`. Load with `peft.PeftModel.from_pretrained(base, repo)` and inject a
**layer-42** residual activation via the Karvonen norm-matched hook (injection char `㈜`, id `158983`),
then generate to read out the lens. (This arm's native source layer is L42.)

Sibling: the L62-trained (mismatch / standard skip-lens) arm is the comparison condition.
"""

api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
with open(os.path.join(SRC, "README.md"), "w") as f:
    f.write(CARD)
api.upload_folder(folder_path=SRC, repo_id=REPO, repo_type="model",
                  ignore_patterns=[".cache*", "*.lock"])
print(f"UPLOADED {REPO} (private)")
