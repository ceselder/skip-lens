"""Create the skip-lens collection (repos already uploaded) with a <150-char description."""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
REPOS = [
    "skip-lens-qwen36-27b-naive-futurelens",
    "skip-lens-qwen36-27b-cnla-warmstart",
    "skip-lens-qwen36-27b-futurelens-rl",
    "skip-lens-qwen36-27b-cnla-rl",
    "skip-lens-qwen36-27b-cnla-longhorizon",
    "skip-lens-qwen36-27b-ar-reconstructor",
    "skip-lens-qwen36-27b-repeatafterme",
    "qwen3.6-27b-nla-av",
    "qwen3.6-27b-nla-rl",
    "qwen3.6-27b-nla-L42",
    "nla-qwen36-27b-matryoshka",
    "oracle-lens-qwen3.6-27b",
]
col = api.create_collection(
    title="Skip-Lens: Multi-Token NLA Lenses (Qwen3.6-27B)", namespace="ceselder",
    description="Multi-token NL activation lenses on Qwen3.6-27B: future-lens, "
                "compositional-NLA warm-start + RL, AR reward model, J-lens NLA.",
    private=False, exists_ok=True)
print("COLLECTION", col.slug, flush=True)
for r in REPOS:
    try:
        api.add_collection_item(col.slug, item_id=f"ceselder/{r}", item_type="model", exists_ok=True)
        print("  added", r, flush=True)
    except Exception as e:
        print("  SKIP", r, repr(e)[:120], flush=True)
print("DONE", col.slug, flush=True)
