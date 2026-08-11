"""Re-target fed-layer readouts to the LOCAL per-layer J-lens workspace.

For each fed layer l, the workspace target becomes topk(lens(J_{l->62}.h_l)) — the J-lens read of
THAT layer's own activation — instead of the fixed L42 reference. Reuses the already-generated
readouts (no regeneration); only recomputes `jlens_top` per (item, fed_layer). Then judge_fedlayer
scores each readout against its local target. At fed L42 the local target == the old fixed one
(jorigin was 42), so those numbers are unchanged; every other depth changes.
"""
import argparse, glob, json, os
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fl_common import lens_topk, decoder_layers

ap = argparse.ArgumentParser()
ap.add_argument("--readouts", nargs="+", required=True)
ap.add_argument("--jdir", required=True)
ap.add_argument("--jtarget", type=int, default=62)
ap.add_argument("--fed-layers", default="62,55,48,42,34,26,18,10")
ap.add_argument("--evals-dir", required=True)
ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
ap.add_argument("--topk", type=int, default=12)
ap.add_argument("--max-items", type=int, default=999)
ap.add_argument("--suffix", default="_local")
args = ap.parse_args()
dev = "cuda"
FED = [int(x) for x in args.fed_layers.split(",")]

tok = AutoTokenizer.from_pretrained(args.base_ckpt)
model = AutoModelForCausalLM.from_pretrained(
    args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
torch.set_grad_enabled(False)

Jmap = {}                                                   # J_{l->62}; L62 = identity
for l in FED:
    if l == args.jtarget:
        continue
    p = os.path.join(args.jdir, f"J_L{l}_to_L{args.jtarget}.npy")
    if os.path.exists(p):
        Jmap[l] = torch.from_numpy(np.load(p)).float().to(dev)
print(f"[local] J for layers {sorted(Jmap)} (+ identity at L{args.jtarget})", flush=True)

grab = {}
layers = decoder_layers(model)
for l in FED:
    layers[l].register_forward_hook(
        lambda m, i, o, L=l: grab.__setitem__(L, (o[0] if isinstance(o, tuple) else o).detach()))

items = []
for f in sorted(glob.glob(os.path.join(args.evals_dir, "lens-eval-*.json"))):
    for it in json.load(open(f))["items"][: args.max_items]:
        items.append({"name": it["name"], "prompt": it["prompt"]})
print(f"[local] {len(items)} items", flush=True)

localtop = {}
for j, it in enumerate(items):
    ids = tok(it["prompt"], return_tensors="pt", truncation=True, max_length=512).input_ids.to(dev)
    ids = ids[:, -256:]
    model(input_ids=ids)
    d = {}
    for l in FED:
        h = grab[l][0, -1].float()
        act = h if l == args.jtarget else (Jmap[l] @ h if l in Jmap else None)
        if act is None:
            continue
        ti, _ = lens_topk(model, act.float(), k=args.topk)
        d[l] = [tok.decode([int(i)]) for i in ti]
    localtop[it["name"]] = d
    if j % 25 == 0:
        print(f"[local] item {j}/{len(items)}", flush=True)

outdir = os.path.dirname(args.readouts[0])
json.dump(localtop, open(os.path.join(outdir, "local_jltop.json"), "w"))

for rf in args.readouts:
    recs = json.load(open(rf))
    n_ok = 0
    keep = []
    for r in recs:
        lt = localtop.get(r["name"], {}).get(r["fed_layer"])
        if lt is None:
            continue                                        # drop layers with no local J (none, given FED)
        r["jlens_top"] = lt
        keep.append(r)
        n_ok += 1
    out = rf.replace(".json", args.suffix + ".json")
    json.dump(keep, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"[local] {os.path.basename(rf)} -> {os.path.basename(out)} ({n_ok}/{len(recs)} re-targeted)", flush=True)
print("REFSCORE_LOCAL_DONE", flush=True)
