"""Upload the key local skip-lens checkpoints to HF + build a collection.

Uploads 6 local adapters as ceselder/skip-lens-qwen36-27b-* repos with model cards,
then creates a collection grouping them WITH the skip-lens/NLA-27B repos already on HF.
(The 72,750-example cNLA warm-start is still training; added later when it finishes.)
"""
import os
from huggingface_hub import HfApi

TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)
CK = "/workspace/cnla/skip-lens/ckpts"
BASE = "Qwen/Qwen3.6-27B"

# (local_path, repo, title, kind, blurb)
UPLOADS = [
    (f"{CK}/futurelens_scaled_L62/iter_0007579", "skip-lens-qwen36-27b-naive-futurelens",
     "Naive future-lens AV (485k)", "peft",
     "Naive future-lens activation-vector (AV) decoder. LoRA adapter on Qwen3.6-27B that, given a "
     "single residual-stream activation from **layer 62** injected at a fixed position (Karvonen "
     "norm-matched injection, char ㈜ / id 158983), generates the text the model was about to produce "
     "next. SFT on **485,000** (activation, future-tokens) pairs from FineFineWeb (5 positions/doc)."),
    (f"{CK}/cnla_av_L62/iter_0000300", "skip-lens-qwen36-27b-cnla-warmstart",
     "Compositional-NLA warm-start SFT (4,850)", "peft",
     "Compositional NLA (cNLA) 4-bullet warm-start. SFT on **4,850** examples whose target is four "
     "sampled future-lens rollouts formatted as bullets (`* c1\\n* c2\\n* c3\\n* c4`). Warm-start for "
     "the cNLA GRPO stage. (A 15× larger 72,750-example warm-start is a sibling repo.)"),
    (f"{CK}/fvecmp_futurelens_b300/iter_000150", "skip-lens-qwen36-27b-futurelens-rl",
     "Future-lens after GRPO RL", "peft",
     "Future-lens after GRPO RL with a **frozen-AR reconstruction (FVE) reward**. Initialized from the "
     "naive future-lens AV; the reward is how well a frozen reconstructor recovers the layer-62 "
     "activation from the lens's generated explanation."),
    (f"{CK}/fvecmp_cnla_b300/iter_000150", "skip-lens-qwen36-27b-cnla-rl",
     "Compositional-NLA after GRPO RL", "peft",
     "Compositional NLA after GRPO RL with the **leave-one-out + threshold FVE composition reward** "
     "(lstsq-optimal composition of the 4 bullets, per-bullet span advantages)."),
    (f"{CK}/cnla_longhorizon/iter_000350", "skip-lens-qwen36-27b-cnla-longhorizon",
     "Long-horizon cNLA GRPO (snapshot)", "peft",
     "Long-horizon compositional-NLA GRPO run (snapshot). Same LOO+threshold FVE reward as cnla-rl, "
     "trained for a much longer horizon."),
    (f"{CK}/ar_abl_anchor/iter_0001500", "skip-lens-qwen36-27b-ar-reconstructor",
     "AR reconstructor / reward model (23.3% FVE)", "ar",
     "The frozen **activation reconstructor (AR)** that serves as the RL reward model. Qwen3.6-27B "
     "truncated to 63 layers + a Linear(5120→5120) value head that reads the backbone's last-token "
     "hidden state at the `</text> <summary>` anchor and reconstructs the **layer-62** activation from "
     "an explanation. **23.3% held-out FVE** (best of the read-position/head ablation). NON-standard "
     "format: `ar_lora_value_head.safetensors` + `ar_meta.json`."),
]


def card(title, blurb, kind):
    if kind == "peft":
        load = ("This is a **PEFT LoRA adapter** (r=64, α=16, rsLoRA) on `Qwen/Qwen3.6-27B`. Load with "
                "`peft.PeftModel.from_pretrained(base, repo)` and inject a layer-62 residual activation "
                "via the Karvonen norm-matched hook at the injection token (char `㈜`, id `158983`), then "
                "generate to read out the lens.")
    else:
        load = ("**Custom AR format** (not standard PEFT): `ar_lora_value_head.safetensors` + "
                "`ar_meta.json`. Rebuild via `nla.train_sft.init_critic_from_base(base, "
                "ar_meta['ar_num_layers'], ...)` → `inject_adapter_in_model(LoraConfig(**ar_meta), "
                "critic.backbone)` → `load_state_dict(...)`. Reads the last-token hidden state and "
                "predicts the layer-62 activation.")
    return f"""---
base_model: {BASE}
library_name: peft
pipeline_tag: text-generation
tags:
- skip-lens
- interpretability
- nla
- future-lens
- activation-oracle
- qwen3.6-27b
license: apache-2.0
---

# {title}

Part of the **skip-lens** project — multi-token natural-language activation lenses on Qwen3.6-27B.
A skip-lens takes a single residual-stream activation and decodes, in natural language, what the
model is *about to say* — and for intermediate layers, surfaces the model's **workspace** content.

{blurb}

**Base model:** `{BASE}` · **LoRA:** r=64, α=16, rsLoRA · **Read/inject layer:** 62.

## Loading

{load}

See the **Skip-Lens** collection for the full family (naive future-lens, compositional-NLA warm-start
+ RL, the AR reconstructor reward model, repeat-after-me, and the J-lens-fitted NLA variants).
"""


for path, repo, title, kind, blurb in UPLOADS:
    rid = f"ceselder/{repo}"
    api.create_repo(rid, repo_type="model", private=False, exist_ok=True)
    with open(os.path.join(path, "README.md"), "w") as f:
        f.write(card(title, blurb, kind))
    api.upload_folder(folder_path=path, repo_id=rid, repo_type="model",
                      ignore_patterns=["reference*", ".cache*", "*.lock"])
    print(f"UPLOADED {rid}", flush=True)

EXISTING = ["skip-lens-qwen36-27b-repeatafterme", "qwen3.6-27b-nla-av", "qwen3.6-27b-nla-rl",
            "qwen3.6-27b-nla-L42", "nla-qwen36-27b-matryoshka", "oracle-lens-qwen3.6-27b"]
col = api.create_collection(
    title="Skip-Lens: Multi-Token NLA Lenses (Qwen3.6-27B)", namespace="ceselder",
    description="Multi-token natural-language activation lenses on Qwen3.6-27B: naive future-lens, "
                "compositional-NLA (4-bullet) warm-start + RL, the frozen AR reconstructor reward model, "
                "repeat-after-me, and J-lens-fitted NLA variants.",
    private=False, exists_ok=True)
print(f"COLLECTION {col.slug}", flush=True)
for r in [u[1] for u in UPLOADS] + EXISTING:
    try:
        api.add_collection_item(col.slug, item_id=f"ceselder/{r}", item_type="model", exists_ok=True)
        print(f"  added {r}", flush=True)
    except Exception as e:
        print(f"  SKIP {r}: {e}", flush=True)
print("ALL_DONE", flush=True)
