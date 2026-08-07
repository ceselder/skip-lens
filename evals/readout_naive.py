"""Read out bullets from a naive-futurelens / AO / RL variant, for judge_intermediates.py.

Each variant is a LoRA adapter that, given a norm-matched activation injected at the injection
token (via the nla Karvonen hook), generates the text the subject model would produce next. We
sample K continuations per eval item; those K continuations ARE the K bullets the intermediate
judge scores (pass@k-bullets).

Two correctness points, mirrored from the interface (weirdchat_lens.py) and readout_inverter.py:

  1. The subject activation is read with the adapter DISABLED (peft.disable_adapter), so the lens
     never reads its own LoRA perturbation. It is the OUTPUT of block `read_layer` at the last
     prompt token -- the exact convention these AVs were trained on.

  2. Feed modes (see evals/naive_variants.json):
       raw : read at the variant's native block, feed h directly.
       jac : read at source block S, feed J_{S->62}.h_S (block-62 basis). Only meaningful for AVs
             whose native block is 62 -- asserted below.

    python evals/readout_naive.py --registry evals/naive_variants.json \
        --ckpt-root <dir of variant subdirs> --jlens-dir <dir of J_L*_to_L62.npy> \
        --nla-path <easynla-ae repo root, importable as `nla`> \
        --variant naive_fl_L62 --feed raw --K 8 --out run/evals/ro__naive_fl_L62__raw.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
DISTS = ["multihop", "multilingual", "order_of_ops", "poetry", "typo", "association"]
SNAP = "Qwen/Qwen3.6-27B"


def find_layers(m):
    best = None
    for _, mod in m.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 32:
            if best is None or len(mod) > len(best):
                best = mod
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(HERE / "naive_variants.json"))
    ap.add_argument("--ckpt-root", required=True, help="dir containing the variant subdirs")
    ap.add_argument("--jlens-dir", required=True, help="dir of J_L{src}_to_L62.npy")
    ap.add_argument("--nla-path", required=True, help="repo root importable as `nla`")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--feed", choices=["raw", "jac", "mismatch"], required=True,
                    help="raw: read at native block. jac: read at --jac-src, apply J_{src->62}. "
                         "mismatch: read raw at --jac-src and feed it straight into the L62 AV "
                         "(no J) -- the test-time layer mismatch.")
    ap.add_argument("--jac-src", type=int, default=0, help="source block for --feed jac/mismatch")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-items", type=int, default=0, help="0 = all")
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    dev = f"cuda:{A.gpu}"
    torch.set_grad_enabled(False)

    reg = json.load(open(A.registry))
    meta, V = reg["_meta"], reg["variants"][A.variant]
    target = int(meta["target_block"])
    if A.feed in ("jac", "mismatch"):
        assert V["native_layer"] == target, (
            f"--feed {A.feed} reads at an off-target block and feeds the block-{target} AV, so "
            f"{A.variant} must be native at {target}; it is native at {V['native_layer']}")
        read_layer = A.jac_src
    else:
        read_layer = int(V["native_layer"])

    ckpt = os.path.join(A.ckpt_root, V["dir"], V["iter"])
    assert os.path.isdir(ckpt), f"missing adapter dir: {ckpt}"

    # nla injection machinery (same as the interface)
    sys.path.insert(0, A.nla_path)
    from nla.utils.hooks import register_karvonen_hook
    from nla.schema import compute_canonical_neighbors
    from nla.datagen.injection_tokens import find_injection_token

    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAP)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = AutoConfig.from_pretrained(SNAP)
    if hasattr(cfg, "language_model_only"):
        cfg.language_model_only = True
    base = AutoModelForCausalLM.from_pretrained(SNAP, config=cfg, dtype=torch.bfloat16,
                                                device_map={"": A.gpu})
    from peft import PeftModel
    PEFT = PeftModel.from_pretrained(base, ckpt, adapter_name="v").eval()

    ACTOR = meta["actor_template"]
    inj_char, inj_id = find_injection_token(tok)
    left, right = compute_canonical_neighbors(tok, ACTOR, inj_char, inj_id)
    vref = [None]
    register_karvonen_hook(PEFT, vref, inj_id, left, right)
    _s = tok.apply_chat_template(
        [{"role": "user", "content": ACTOR.format(injection_char=inj_char)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    PT = torch.tensor([tok.encode(_s, add_special_tokens=False)], device=dev)

    J = None
    if A.feed == "jac":
        jp = os.path.join(A.jlens_dir, f"J_L{A.jac_src}_to_L{target}.npy")
        assert os.path.exists(jp), f"missing Jacobian: {jp}"
        J = torch.from_numpy(np.load(jp)).to(dev).float()

    layers = find_layers(PEFT)
    CAP = {}
    layers[read_layer].register_forward_hook(
        lambda m, i, o: CAP.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))

    @torch.no_grad()
    def bullets(text):
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        with PEFT.disable_adapter():            # clean subject activation (no LoRA perturbation)
            PEFT(input_ids=ids, use_cache=False)
        h = CAP["h"][0, -1].float()
        if J is not None:
            h = J @ h                            # -> block-62 basis
        PEFT.set_adapter("v")
        vref[0] = h.view(1, -1).expand(A.K, -1).contiguous()
        try:
            g = PEFT.generate(PT.repeat(A.K, 1), attention_mask=torch.ones(A.K, PT.shape[1],
                              device=dev, dtype=torch.long), do_sample=A.temp > 0,
                              temperature=max(A.temp, 1e-5), top_p=0.95,
                              min_new_tokens=A.max_new, max_new_tokens=A.max_new,
                              pad_token_id=tok.pad_token_id)
        finally:
            vref[0] = None
        outs = [tok.decode(g[b][PT.shape[1]:], skip_special_tokens=True).strip()
                for b in range(A.K)]
        return [o.replace("\n", " ").strip() for o in outs if o.strip()]

    R, n = {}, 0
    for name in DISTS:
        p = HERE / "datasets" / f"{name}.json"
        if not p.exists():
            continue
        items = json.load(open(p))["items"]
        if A.max_items:
            items = items[:A.max_items]
        R[name] = {}
        for i, it in enumerate(items):
            try:
                R[name][str(i)] = bullets(it["prompt"])
            except Exception as e:
                print(f"[ro] {A.variant}/{A.feed} {name}[{i}] failed: {type(e).__name__}: {e}",
                      flush=True)
                continue
            n += 1
            if n % 40 == 0:
                print(f"[ro] {A.variant}/{A.feed}: {n} items", flush=True)
        print(f"[ro] {name}: {len(R[name])}/{len(items)}", flush=True)

    Path(os.path.dirname(A.out)).mkdir(parents=True, exist_ok=True)
    # top-level {dist: {idx: [bullets]}} so judge_intermediates.py consumes it directly;
    # the judge only looks up the 6 dist names, so "_meta" is ignored by it.
    out_obj = dict(R)
    out_obj["_meta"] = {"variant": A.variant, "feed": A.feed,
                        "jac_src": A.jac_src if A.feed == "jac" else None,
                        "read_layer": read_layer, "K": A.K}
    json.dump(out_obj, open(A.out, "w"), indent=1)
    ex = next((v["0"] for v in R.values() if "0" in v), None)
    print(f"[ro] wrote {n} read-outs -> {A.out}")
    if ex:
        print("example bullets:\n  " + "\n  ".join(ex[:5]))
    print("RO_DONE")


if __name__ == "__main__":
    main()
