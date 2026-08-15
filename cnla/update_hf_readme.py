"""Prepend a scale-test warning banner to the HF model card."""
import os
from huggingface_hub import HfApi, hf_hub_download

REPO = "ceselder/skip-lens-qwen36-27b-meansub-norm"
TOKEN = os.environ["HF_TOKEN"]
cur = open(hf_hub_download(REPO, "README.md", token=TOKEN)).read()
banner = (
    "> **UPDATE (scale test) — this variant is NOT a real improvement over plain injection.**\n"
    "> The +0.119 @ L42 was a *small-data artifact*: it only rescued an undertrained 500-pair raw lens.\n"
    "> Retrained at 5k pairs, plain raw injection catches up exactly (raw **0.511** vs mean-centered\n"
    "> **0.508**, gap **-0.003**), and mean-centering *hurts* at L62 (raw 0.72 vs mc 0.66). **The real\n"
    "> lever is training data, not mean-centering** (raw scales 0.41->0.51->0.58 at 500->5k->150k).\n"
    "> Use only as a small-data curiosity; for a better lens, train raw on more data. See report §5c.\n\n"
)
parts = cur.split("---", 2)
new = f"---{parts[1]}---\n\n{banner}{parts[2].lstrip()}" if len(parts) == 3 else banner + cur
HfApi(token=TOKEN).upload_file(path_or_fileobj=new.encode(), path_in_repo="README.md",
                               repo_id=REPO, repo_type="model")
print("README_UPDATED")
