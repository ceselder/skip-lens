"""Collect the CSPRNG repeat-after-me control for Skip Lens.

Random words are drawn uniformly with ``secrets.SystemRandom`` from the BIP-39
English word list.  They are *not* turned into checksummed mnemonics or wallet
keys.  We keep only phrases the unadapted model greedily repeats token-exactly,
then harvest aligned middle/late residuals from its own completion.

Each example starts after at least one generated token, so predicting the
remaining 1--16 tokens requires the copy/induction computation rather than a
semantic language prior.  Phrase IDs, not rows, define the train/validation
split, preventing another position from the same random phrase leaking across.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import secrets
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from mnemonic import Mnemonic
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.datagen.injection_tokens import find_injection_token
from nla.schema import compute_canonical_neighbors
from nla.utils.arch_adapters import resolve_decoder_layers
from pretrain.finalize_naive_data import ACTOR_TEMPLATE


def canonical_actor_prompt():
    return [{"role": "user", "content": ACTOR_TEMPLATE.format(injection_char="<INJECT>")}]


def normalized_words(text: str) -> str:
    return " ".join(text.strip().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--n-phrases", type=int, default=700)
    ap.add_argument("--phrase-words", type=int, default=40)
    ap.add_argument("--positions-per-phrase", type=int, default=4)
    ap.add_argument("--max-span", type=int, default=16)
    ap.add_argument("--fixed-span", type=int, default=0,
                    help="if positive, every target has exactly this many tokens")
    ap.add_argument("--layers", type=int, nargs="+", default=[42, 62])
    ap.add_argument("--target-layer", type=int, default=62)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-attempt-multiplier", type=int, default=5)
    args = ap.parse_args()
    if args.target_layer not in args.layers:
        raise ValueError("--target-layer must be included in --layers")
    if args.fixed_span < 0 or args.fixed_span > args.phrase_words - 1:
        raise ValueError("--fixed-span must be 0 or less than --phrase-words")

    device = "cuda"
    rng = secrets.SystemRandom()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to(device).eval()
    torch.set_grad_enabled(False)

    # Uniform over a fixed public dictionary, restricted to words that remain
    # one token both initially and after a space. This makes "16 tokens" exact.
    words = []
    for word in Mnemonic("english").wordlist:
        if (len(tokenizer.encode(word, add_special_tokens=False)) == 1
                and len(tokenizer.encode(" " + word, add_special_tokens=False)) == 1):
            words.append(word)
    if len(words) < args.phrase_words:
        raise RuntimeError(f"only {len(words)} single-token BIP-39 words survived")
    print(f"[words] {len(words)}/2048 BIP-39 words are token-exact", flush=True)

    layers = resolve_decoder_layers(model)
    captured = {}
    for layer in args.layers:
        layers[layer].register_forward_hook(
            lambda _m, _i, out, layer=layer: captured.__setitem__(
                layer, (out[0] if isinstance(out, tuple) else out).detach()))

    accepted = []
    attempts = 0
    max_attempts = args.n_phrases * args.max_attempt_multiplier
    while len(accepted) < args.n_phrases and attempts < max_attempts:
        batch = []
        for _ in range(min(args.batch_size, args.n_phrases - len(accepted))):
            phrase_words = rng.sample(words, args.phrase_words)
            phrase = " ".join(phrase_words)
            user = (
                "Repeat the following random words exactly, in the same order. "
                "Output only the words and nothing else.\n\n" + phrase
            )
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
            expected_ids = tokenizer.encode(phrase, add_special_tokens=False)
            if len(expected_ids) != args.phrase_words:
                continue
            batch.append((phrase, prompt_ids, expected_ids))
        attempts += len(batch)
        if not batch:
            continue

        enc = tokenizer.pad(
            {"input_ids": [x[1] for x in batch]}, padding=True,
            return_tensors="pt").to(device)
        generated = model.generate(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask,
            max_new_tokens=args.phrase_words + 2, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        suffix = generated[:, enc.input_ids.shape[1]:]
        for i, (phrase, prompt_ids, expected_ids) in enumerate(batch):
            got = suffix[i, : len(expected_ids)].tolist()
            if got != expected_ids:
                continue
            accepted.append((phrase, prompt_ids, expected_ids))
            if len(accepted) >= args.n_phrases:
                break
        print(f"[generate] attempts={attempts} token-exact={len(accepted)}", flush=True)

    if len(accepted) < args.n_phrases:
        raise RuntimeError(
            f"only {len(accepted)}/{args.n_phrases} phrases repeated token-exactly "
            f"after {attempts} attempts")

    rows = []
    for b0 in range(0, len(accepted), args.batch_size):
        chunk = accepted[b0:b0 + args.batch_size]
        full = [pids + out_ids for _, pids, out_ids in chunk]
        tokenizer.padding_side = "right"
        enc = tokenizer.pad({"input_ids": full}, padding=True, return_tensors="pt").to(device)
        model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=False)
        for i, (phrase, prompt_ids, output_ids) in enumerate(chunk):
            phrase_id = hashlib.sha256(phrase.encode()).hexdigest()[:20]
            if args.fixed_span:
                max_anchor = len(output_ids) - args.fixed_span
                valid_anchors = range(1, max_anchor + 1)
            else:
                max_anchor = len(output_ids) - 1
                valid_anchors = range(1, max_anchor)
            if len(valid_anchors) < args.positions_per_phrase:
                raise ValueError("not enough valid anchors for --positions-per-phrase")
            anchors = sorted(rng.sample(valid_anchors, args.positions_per_phrase))
            for anchor in anchors:
                horizon = args.fixed_span or rng.randint(
                    1, min(args.max_span, len(output_ids) - anchor)
                )
                absolute_pos = len(prompt_ids) + anchor - 1
                row = {
                    "prompt": canonical_actor_prompt(),
                    "activation_vector": captured[args.target_layer][i, absolute_pos]
                    .float().cpu().tolist(),
                    "teacher_input_ids": (prompt_ids + output_ids[:anchor]),
                    "target_ids": output_ids[anchor:anchor + horizon],
                    "continuation_ids": output_ids[anchor:anchor + horizon],
                    "ctx_text": tokenizer.decode((prompt_ids + output_ids[:anchor])[-96:]),
                    "doc_id": phrase_id,
                    "phrase": phrase,
                    "anchor": anchor,
                    "span_tokens": horizon,
                }
                for layer in args.layers:
                    row[f"act_L{layer}"] = captured[layer][i, absolute_pos].float().cpu().tolist()
                rows.append(row)
        tokenizer.padding_side = "left"
        print(f"[capture] {min(b0 + len(chunk), len(accepted))}/{len(accepted)}", flush=True)

    # Split on cryptographically random phrase identity, never individual rows.
    phrase_ids = sorted({r["doc_id"] for r in rows})
    n_val = max(1, round(len(phrase_ids) * args.val_frac))
    val_ids = set(phrase_ids[:n_val])
    prompt_type = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))

    def table(items):
        cols = {
            "prompt": pa.array([r["prompt"] for r in items], type=prompt_type),
            "activation_vector": pa.array(
                [r["activation_vector"] for r in items], type=pa.list_(pa.float32())),
            "teacher_input_ids": pa.array(
                [r["teacher_input_ids"] for r in items], type=pa.list_(pa.int32())),
            "target_ids": pa.array([r["target_ids"] for r in items], type=pa.list_(pa.int32())),
            "response": pa.array(
                [tokenizer.decode(r["target_ids"]) for r in items], type=pa.string()),
            "continuation_ids": pa.array(
                [r["continuation_ids"] for r in items], type=pa.list_(pa.int32())),
            "ctx_text": pa.array([r["ctx_text"] for r in items], type=pa.string()),
            "doc_id": pa.array([r["doc_id"] for r in items], type=pa.string()),
            "phrase": pa.array([r["phrase"] for r in items], type=pa.string()),
            "anchor": pa.array([r["anchor"] for r in items], type=pa.int16()),
            "span_tokens": pa.array([r["span_tokens"] for r in items], type=pa.int16()),
        }
        for layer in args.layers:
            cols[f"act_L{layer}"] = pa.array(
                [r[f"act_L{layer}"] for r in items], type=pa.list_(pa.float32()))
        return pa.table(cols)

    train = table([r for r in rows if r["doc_id"] not in val_ids])
    val = table([r for r in rows if r["doc_id"] in val_ids])
    for path, split in ((args.out_train, train), (args.out_val, val)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(split, path, row_group_size=1000)

    inj_char, inj_id = find_injection_token(tokenizer)
    left, right = compute_canonical_neighbors(tokenizer, ACTOR_TEMPLATE, inj_char, inj_id)
    sidecar = {
        "dataset_id": f"repeat_csprng_{args.base_model.split('/')[-1]}_L{args.target_layer}",
        "stage": "opd", "row_count": train.num_rows,
        "kind": "nla_dataset", "schema_version": 1,
        "extraction": {
            "base_model": args.base_model,
            "d_model": model.config.get_text_config().hidden_size,
            "layer_index": args.target_layer, "norm": "none",
        },
        "tokens": {
            "injection_char": inj_char, "injection_token_id": inj_id,
            "injection_left_neighbor_id": left,
            "injection_right_neighbor_id": right,
            "critic_suffix_ids": None,
        },
        "prompt_templates": {"actor": ACTOR_TEMPLATE},
        "randomness": {
            "generator": "secrets.SystemRandom (OS CSPRNG)",
            "dictionary": "BIP-39 English, uniform without replacement per phrase",
            "wallet_mnemonics": False,
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "created_by": "pretrain.collect_repeat_data",
    }
    for path in (args.out_train, args.out_val):
        Path(path + ".nla_meta.yaml").write_text(
            yaml.safe_dump(sidecar, sort_keys=False, allow_unicode=True))
    manifest = {
        "accepted_phrases": len(accepted), "attempted_phrases": attempts,
        "train_rows": train.num_rows, "val_rows": val.num_rows,
        "max_span": args.max_span, "fixed_span": args.fixed_span or None,
        "phrase_set_sha256": hashlib.sha256(
            "\n".join(sorted(x[0] for x in accepted)).encode()).hexdigest(),
    }
    Path(args.out_train + ".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
