"""Upload the L62-trained skip-lens (the non-matched / mismatch arm) to HF — the pair-partner of
ceselder/skip-lens-qwen36-27b-L42-matched for the source-layer ablation."""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
SRC = "/workspace/cnla/skip-lens/ckpts/skiplens_L62_150k/iter_0002344"
REPO = "ceselder/skip-lens-qwen36-27b-L62-mismatch"
SIBLING = "ceselder/skip-lens-qwen36-27b-L42-matched"
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

# Skip-Lens · source-layer ablation · L62 (MISMATCH / standard skip-lens) arm

The **non-matched** arm of the skip-lens source-layer ablation on Qwen3.6-27B — i.e. the **standard
skip-lens**. A future-lens activation-decoder (LoRA r64 α16 rsLoRA, all modules) trained by AV-SFT to
reconstruct the on-policy future continuation from an injected residual, with the **source layer =
block 62 (penultimate)**. Called the "mismatch" arm because at test time you feed it an *intermediate*
(e.g. L42) — a layer ≠ its training layer — to surface the model's hidden workspace.

- **Data:** the SAME 150k on-policy FineWeb continuations as the matched-L42 sibling (`fl_big` subset).
  Token-matched — the *only* difference from [`{SIBLING}`](https://huggingface.co/{SIBLING}) is the
  source layer whose activation is injected during training (L62 here vs L42 there).
- **Recipe:** `train_sft --mode av`, effective batch 16×4=64, 1 epoch (2,344 steps). Verbatim the
  canonical futurelens recipe.
- **Result (fed raw L42, disagreement set, Sonnet-5 judge):** reads the workspace (J-lens top-k
  concepts) at **agree_jlens 0.65** vs the matched-L42 arm's 0.61, and **leaks the surface answer less**
  (agree_answer **0.33** vs 0.39). Its answer-agreement decays monotonically as you feed shallower —
  the faithful-intermediate signature. So the train/feed **mismatch keeps the lens from collapsing into
  an answer-decoder**.

## Loading

PEFT LoRA adapter on `{BASE}`. Load with `peft.PeftModel.from_pretrained(base, repo)` and inject a
residual activation via the Karvonen norm-matched hook (injection char `㈜`, id `158983`). Feed an
**intermediate** layer (e.g. L42) to read the workspace; feed L62 for its native (answer-heavy) readout.

**Pair:** [`{SIBLING}`](https://huggingface.co/{SIBLING}) — same corpus, source layer 42 (matched).
"""

api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
with open(os.path.join(SRC, "README.md"), "w") as f:
    f.write(CARD)
api.upload_folder(folder_path=SRC, repo_id=REPO, repo_type="model",
                  ignore_patterns=[".cache*", "*.lock"])
print(f"UPLOADED {REPO} (private)")
