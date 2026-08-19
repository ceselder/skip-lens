"""Official multi-token workspace readout evaluation.

Uses Anthropic's released methodological prompt sets, their specified read
positions, and a layer sweep.  For each activation it records:

* raw Skip-Lens text from h_l;
* Jacobian-transported Skip-Lens text from J_{l->62} h_l;
* a shuffled-activation hallucination control (optional);
* the ordinary J-lens top tokens from the same transported vector.

The companion judge scores semantic concept coverage, joint multi-concept
coverage in one coherent readout, answer-skipping and unrelated hallucination.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.utils import register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers, resolve_text_model
from nla.utils.prompts import build_prompt_text

NUMBER_WORDS = {
    "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
    "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
    "12": "twelve", "13": "thirteen", "15": "fifteen", "16": "sixteen",
    "20": "twenty", "24": "twenty-four",
}
OPERATION_VARIANTS = {
    "addition": ["addition", "add", "plus", "+"],
    "subtraction": ["subtraction", "subtract", "minus", "-"],
    "multiplication": ["multiplication", "multiply", "times", "*", "×"],
    "division": ["division", "divide", "divided", "/", "÷"],
    "mod": ["mod", "modulo", "remainder", "%"],
    "squared": ["squared", "square", "²"],
}


def read_position(tokenizer, prompt: str, distribution: str) -> int:
    """Resolve the paper-specified activation position in full-prompt tokens."""
    ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if distribution != "poetry":
        return len(ids) - 1
    boundary = prompt.rfind("\n") + 1
    if boundary <= 0:
        raise ValueError("official poetry item has no newline")
    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.offset_mapping
    candidates = [i for i, (_start, end) in enumerate(offsets) if end <= boundary]
    if not candidates:
        raise ValueError("could not align poetry newline to a token")
    pos = candidates[-1]
    decoded_prefix = tokenizer.decode(encoded.input_ids[:pos + 1])
    if not decoded_prefix.endswith("\n"):
        raise ValueError(
            f"poetry read position did not end at newline: {decoded_prefix[-40:]!r}")
    return pos


def distribution_name(path: str) -> str:
    name = Path(path).stem.removeprefix("lens-eval-")
    return "order_ops" if name == "order-ops" else name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--av-ckpt", required=True)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--jdir", required=True)
    ap.add_argument("--datasets-dir", default="evals/datasets/official/evaluations")
    ap.add_argument("--layers", default="20,26,32,38,42,48,54,62")
    ap.add_argument("--target-layer", type=int, default=62)
    ap.add_argument("--modes", default="raw,jac,shuffle")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--generation-batch", type=int, default=8)
    ap.add_argument("--jlens-topk", type=int, default=20)
    ap.add_argument("--max-items-per-dist", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    layers_to_read = [int(x) for x in args.layers.split(",")]
    modes = args.modes.split(",")
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = load_nla_config(args.sidecar, tok)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    model = PeftModel.from_pretrained(base, args.av_ckpt).eval()
    vref = [None]
    register_karvonen_hook(
        model, vref, cfg.injection_token_id,
        cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id)
    decoder = resolve_decoder_layers(model.get_base_model())
    invalid_layers = [layer for layer in layers_to_read
                      if not 0 <= layer < len(decoder)]
    if invalid_layers:
        raise ValueError(
            f"decoder has {len(decoder)} blocks; invalid module indices {invalid_layers}"
        )
    captured = {}
    for layer in layers_to_read:
        decoder[layer].register_forward_hook(
            lambda _m, _i, out, layer=layer: captured.__setitem__(
                layer, (out[0] if isinstance(out, tuple) else out).detach()))

    jacobians = {}
    for layer in layers_to_read:
        if layer == args.target_layer:
            continue
        path = os.path.join(args.jdir, f"J_L{layer}_to_L{args.target_layer}.npy")
        if os.path.exists(path):
            jacobians[layer] = torch.from_numpy(np.load(path)).float().to(device)
        elif "jac" in modes:
            print(f"[warn] missing {path}; jac mode will omit L{layer}", flush=True)

    # Load the exact upstream schema and capture clean activations at its exact positions.
    items = []
    for path in sorted(glob.glob(os.path.join(args.datasets_dir, "lens-eval-*.json"))):
        dist = distribution_name(path)
        values = json.loads(Path(path).read_text())["items"]
        if args.max_items_per_dist:
            values = values[:args.max_items_per_dist]
        for item in values:
            prompt = item["prompt"]
            pos = read_position(tok, prompt, dist)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            with torch.no_grad(), model.disable_adapter():
                model(input_ids=ids, use_cache=False)
            hs = {layer: captured[layer][0, pos].float().cpu() for layer in layers_to_read}
            items.append({
                "distribution": dist, "name": item["name"], "prompt": prompt,
                "target": item.get("target"),
                "intermediates": item["intermediates"], "read_position": pos,
                "activations": hs,
            })
        print(f"[capture] {dist}: {len(values)}", flush=True)

    text_model = resolve_text_model(model.get_base_model())
    final_norm = text_model.model.norm
    unembed = model.get_output_embeddings()

    def mapped(item, layer):
        h = item["activations"][layer].to(device)
        if layer == args.target_layer:
            return h
        J = jacobians.get(layer)
        return (J @ h) if J is not None else None

    def top_tokens(vector):
        with torch.no_grad():
            logits = unembed(final_norm(vector.to(unembed.weight.dtype))).float()
        ids = logits.topk(args.jlens_topk).indices.tolist()
        return [tok.decode([i]) for i in ids], ids

    def jlens_covered(intermediates, token_ids):
        top = set(token_ids)
        covered = []
        for concept in intermediates:
            variants = [concept]
            variants.extend(OPERATION_VARIANTS.get(concept.lower(), []))
            if concept in NUMBER_WORDS:
                variants.append(NUMBER_WORDS[concept])
            ids = set()
            for variant in variants:
                for surface in (variant, " " + variant):
                    encoded = tok.encode(surface, add_special_tokens=False)
                    if len(encoded) == 1:
                        ids.add(encoded[0])
            if ids & top:
                covered.append(concept)
        return covered

    actor_text = build_prompt_text(items[0].get("prompt_msgs", [
        {"role": "user", "content": cfg.actor_prompt_template.format(
            injection_char="<INJECT>")}
    ]), cfg.injection_char, tok)
    # The fallback above creates an already-substituted marker inside a chat
    # message; build_prompt_text is retained so chat-template behavior exactly
    # matches training.
    actor_ids = tok.encode(actor_text, add_special_tokens=False)

    requests = []
    for item_idx, item in enumerate(items):
        for layer in layers_to_read:
            jh = mapped(item, layer)
            for mode in modes:
                if mode == "raw":
                    vector = item["activations"][layer]
                    jlens_vector = jh
                elif mode == "jac":
                    vector = jh.cpu() if jh is not None else None
                    jlens_vector = jh
                elif mode == "shuffle":
                    # Deterministic within-distribution mismatch; preserves layer/norm.
                    peers = [i for i, x in enumerate(items)
                             if x["distribution"] == item["distribution"]]
                    peer = peers[(peers.index(item_idx) + 1) % len(peers)]
                    vector = items[peer]["activations"][layer]
                    jlens_vector = mapped(items[peer], layer)
                else:
                    raise ValueError(f"unknown mode {mode}")
                if vector is None:
                    continue
                if jlens_vector is not None:
                    jlens_strings, jlens_ids = top_tokens(jlens_vector)
                else:
                    jlens_strings, jlens_ids = [], []
                requests.append({"item_idx": item_idx, "layer": layer, "mode": mode,
                                 "vector": vector,
                                 "jlens_top": jlens_strings,
                                 "jlens_covered": jlens_covered(
                                     item["intermediates"], jlens_ids),
                                 })

    records = []
    expanded = []
    for request_idx, request in enumerate(requests):
        for sample in range(args.samples):
            expanded.append((request_idx, sample, request))
    generations = {i: [] for i in range(len(requests))}
    prompt_tensor = torch.tensor(actor_ids, dtype=torch.long, device=device)
    for b0 in range(0, len(expanded), args.generation_batch):
        chunk = expanded[b0:b0 + args.generation_batch]
        vectors = torch.stack([x[2]["vector"].float() for x in chunk]).to(device)
        ids = prompt_tensor.view(1, -1).repeat(len(chunk), 1)
        vref[0] = vectors
        try:
            with torch.no_grad():
                gen = model.generate(
                    input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    **({"temperature": args.temperature, "top_p": 0.95, "top_k": 0}
                       if args.temperature > 0 else {}),
                    pad_token_id=tok.eos_token_id)
        finally:
            vref[0] = None
        for row, (request_idx, _sample, _request) in zip(gen, chunk):
            generations[request_idx].append(
                tok.decode(row[len(actor_ids):], skip_special_tokens=True).strip())
        if b0 % (args.generation_batch * 20) == 0:
            print(f"[generate] {b0}/{len(expanded)}", flush=True)

    for request_idx, request in enumerate(requests):
        item = items[request["item_idx"]]
        records.append({
            "distribution": item["distribution"], "name": item["name"],
            "prompt": item["prompt"], "target": item["target"],
            "intermediates": item["intermediates"],
            "read_position": item["read_position"],
            "layer": request["layer"], "mode": request["mode"],
            "readouts": generations[request_idx],
            "jlens_top": request["jlens_top"],
            "jlens_covered": request["jlens_covered"],
        })

    out = {
        "meta": {
            "base": args.base_ckpt, "checkpoint": args.av_ckpt,
            "official_datasets": True, "layers": layers_to_read,
            "modes": modes, "samples": args.samples, "seed": args.seed,
            "activation_indexing": (
                "layer K = output of decoder.layers[K] = HF hidden_states[K+1]; "
                "raw mode injects this post-block residual without Jacobian transport"
            ),
            "decoder_blocks": len(decoder),
        },
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}: {len(records)} records / {len(expanded)} readouts")


if __name__ == "__main__":
    main()
