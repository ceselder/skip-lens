"""Pre-populate the vibecoded interface's eval-builder (eval_categories.json) with the 6 paper
intermediate-surfacing datasets, so the user can build 'coherent surfacing + no hallucination'
eval categories on the paper's activation/settings pairs. Merges (never overwrites existing)."""
import json, glob, os
from transformers import AutoTokenizer

DATASETS = "/workspace-vast/celeste/cnla-skip-lens/evals/datasets"
CATS = "/workspace-vast/celeste/regression-futurelens/eval_categories.json"
JUDGE = (
    "You are judging an interpretability lens readout of a language model's internal activation. "
    "Answer YES if the readout COHERENTLY surfaces the target concept (or a clear paraphrase) and "
    "does NOT hallucinate substantial unrelated content; answer NO otherwise.\n\n"
    "Target concept: {concept}\n\nLens readout:\n{readout}\n\nReply with only YES or NO."
)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
try:
    d = json.load(open(CATS))
except Exception:
    d = {"categories": []}
existing = {c["name"] for c in d["categories"]}

added = 0
for f in sorted(glob.glob(os.path.join(DATASETS, "*.json"))):
    name = os.path.basename(f)[:-5]
    if name == "manifest":
        continue
    cname = "paper:" + name
    if cname in existing:
        print("skip", cname, "(exists)")
        continue
    data = json.load(open(f))
    items = data.get("items") if isinstance(data, dict) else data
    exs = []
    for it in items[:30]:
        prompt = it["prompt"] if isinstance(it, dict) else it
        inter = it.get("intermediates", []) if isinstance(it, dict) else []
        concept = ""
        if inter:
            concept = inter[0].get("accept", [None])[0] or inter[0].get("name", "")
        ids = tok(prompt, truncation=True, max_length=512).input_ids
        exs.append({"ctx_text": prompt, "layer": 62, "pos": len(ids) - 1,
                    "ground_truth": "YES", "concept": concept,
                    "token": tok.decode([ids[-1]])})
    d["categories"].append({"name": cname, "prompt": JUDGE, "examples": exs})
    added += 1
    eg = exs[0]["concept"] if exs else ""
    print("added", cname, "-", len(exs), "examples | e.g. concept:", repr(eg))

json.dump(d, open(CATS, "w"), indent=2, ensure_ascii=False)
print("[done]", added, "paper categories ->", CATS, "| total now", len(d["categories"]))
