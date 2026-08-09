"""Collect AO / self-explainer training data, rollout-grounded.

For each sampled position t in a corpus doc, store:
  - activation_vector : penultimate (block-62) residual at t.
  - rollouts          : K sampled continuations the model ACTUALLY produces from
                        the context up to t (grounds the label in behavior — the
                        "take a bunch of rollouts to check what it's about to say"
                        anti-confabulation move).
  - top_tokens        : top-k next-token strings (the distribution) — cheap extra
                        signal for the labeler.
  - ctx_text          : decoded context tail (for eval display).
  - teacher_input_ids : exact source tokens through t.  OPD gives these to the
                        frozen teacher while the student receives only the
                        activation.  Token IDs avoid decode/re-encode drift.
  - continuation_ids  : the corpus continuation after t (evaluation only).
  - rollout_token_ids : exact IDs corresponding to `rollouts` (SFT targets).
  - prompt            : the AV inject template with <INJECT>.

Claude then labels each row (gen_concept_descriptions / a rollout-summary prompt)
-> `response` column, and train_sft --mode av trains the AO (inject at block 1).
"""
import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.datagen.injection_tokens import find_injection_token
from nla.schema import compute_canonical_neighbors

ACTOR_TEMPLATE = (
    "You are shown an internal activation vector captured from a language model "
    "as it reads a passage of text. The vector, enclosed in <concept> tags, is "
    "taken at one position and encodes what the model is about to generate next. "
    "Output the text the model most likely produces immediately after this point.\n\n"
    "<concept>{injection_char}</concept>")
INJECT_PLACEHOLDER = "<INJECT>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpus-config", default=None,
                    help="optional Hugging Face dataset config when --corpus is a Hub ID")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=62)
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="multiple TRAINING layers to grab, token-matched (overrides --layer); "
                         "stored as act_L{l} columns for a training-layer sweep")
    ap.add_argument("--n-docs", type=int, default=700)
    ap.add_argument("--doc-offset", type=int, default=0,
                    help="skip the first N corpus docs (for sharded parallel collection: "
                         "shard i uses --doc-offset i*n_docs --n-docs n_docs, non-overlapping)")
    ap.add_argument("--positions-per-doc", type=int, default=5)
    ap.add_argument("--rollouts", type=int, default=16)   # "a ton" — dense sample of what comes next
    ap.add_argument("--rollout-len", type=int, default=8)  # SHORT — the concept within the next 4-8 tokens
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--min-ctx", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--decision-points", action=argparse.BooleanOptionalAction, default=True,
                    help="select high-entropy CONTENT decision points (many possible continuations) "
                         "instead of uniform-random positions")
    ap.add_argument("--select", choices=["entropy", "agreement"], default="entropy",
                    help="entropy = high-entropy decision points; agreement = where J-lens & tuned-lens "
                         "predictions agree (faithful next-verbalization spots)")
    ap.add_argument("--jdir", default=None, help="J-lens matrices dir (for --select agreement)")
    ap.add_argument("--tuned-lens", default=None, help="tuned lens .pt (for --select agreement)")
    ap.add_argument("--agree-layer", type=int, default=42, help="layer to score J/tuned agreement at")
    ap.add_argument("--target-layer", type=int, default=62, help="J-lens target (penultimate)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    LAYERS = args.layers if args.layers else [args.layer]
    torch.manual_seed(args.seed)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.base_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    layers = (model.model if hasattr(model, "model") else model).layers
    grabbed = {}
    def _mk(l):
        return lambda m, i, o: grabbed.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach())
    for _l in LAYERS:
        layers[_l].register_forward_hook(_mk(_l))

    # agreement-selection machinery: J-lens + tuned-lens readouts at agree-layer
    grab2, J_ag, tA, tb, W_U, fnorm = {}, None, None, None, None, None
    if args.select == "agreement":
        bc = model.model if hasattr(model, "model") else model
        fnorm, W_U = bc.norm, model.get_output_embeddings()
        J_ag = torch.from_numpy(np.load(os.path.join(
            args.jdir, f"J_L{args.agree_layer}_to_L{args.target_layer}.npy"))).float().to(dev)
        tl = torch.load(args.tuned_lens, map_location=dev); tA, tb = tl["A"].float().to(dev), tl["b"].float().to(dev)
        layers[args.agree_layer].register_forward_hook(
            lambda m, i, o: grab2.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
        print(f"[agreement] J_L{args.agree_layer}->L{args.target_layer} + tuned lens loaded", flush=True)

    need = args.doc_offset + args.n_docs
    if os.path.exists(args.corpus):
        pf = pq.ParquetFile(args.corpus)
        texts = []
        for rg in range(pf.num_row_groups):
            texts.extend(pf.read_row_group(rg, columns=["text"]).column("text").to_pylist())
            if len(texts) >= need:
                break
        texts = texts[args.doc_offset:need]
    else:
        from datasets import load_dataset
        stream = load_dataset(args.corpus, args.corpus_config, split="train", streaming=True)
        rows = itertools.islice(stream, args.doc_offset, need)
        texts = [row["text"] for row in rows]
    # This shard's non-overlapping slice.
    texts = [t for t in texts if t and len(t) > 120]
    prompt_msgs = [{"role": "user", "content": ACTOR_TEMPLATE.format(injection_char=INJECT_PLACEHOLDER)}]

    acts = {l: [] for l in LAYERS}
    rolls, roll_ids, tops, ctxs, teacher_ids, continuation_ids = [], [], [], [], [], []
    prompts, docids, ents = [], [], []
    # constant metadata + an atomic writer, so a long/killed collect keeps partial
    # data (this collector otherwise only writes once at the very end).
    inj_char, inj_id = find_injection_token(tok)
    left, right = compute_canonical_neighbors(tok, ACTOR_TEMPLATE, inj_char, inj_id)
    d_model = model.config.get_text_config().hidden_size
    ps = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))

    def write_out():
        cols = {
            "prompt": pa.array(prompts, type=ps),
            "rollouts": pa.array(rolls, type=pa.list_(pa.string())),
            "rollout_token_ids": pa.array(
                roll_ids, type=pa.list_(pa.list_(pa.int32()))),
            "top_tokens": pa.array(tops, type=pa.list_(pa.string())),
            "ctx_text": pa.array(ctxs, type=pa.string()),
            "teacher_input_ids": pa.array(
                teacher_ids, type=pa.list_(pa.int32())),
            "continuation_ids": pa.array(
                continuation_ids, type=pa.list_(pa.int32())),
            "doc_id": pa.array(docids, type=pa.string()),
            "next_token_entropy": pa.array(ents, type=pa.float32()),
        }
        for l in LAYERS:
            cols[f"act_L{l}"] = pa.array(acts[l], type=pa.list_(pa.float32()))
        cols["activation_vector"] = pa.array(acts[LAYERS[0]], type=pa.list_(pa.float32()))
        table = pa.table(cols)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out + ".tmp"
        pq.write_table(table, tmp, row_group_size=2000)
        os.replace(tmp, args.out)
        Path(args.out + ".meta.json").write_text(json.dumps(
            {"inj_char": inj_char, "inj_id": inj_id, "left": left, "right": right,
             "d_model": d_model, "layer": LAYERS[0], "layers": LAYERS, "rows": table.num_rows}))
        return table.num_rows
    for di, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length).input_ids.to(dev)
        L = ids.shape[1]
        if L < args.min_ctx + args.rollout_len + 2:
            continue
        with torch.no_grad():
            out = model(input_ids=ids)
        hs = {l: grabbed[l][0].float() for l in LAYERS}   # {training-layer: [L, d]}
        logits = out.logits[0].float()        # [L, vocab]
        hi = L - args.rollout_len - 1
        if hi <= args.min_ctx:
            continue
        n = min(args.positions_per_doc, hi - args.min_ctx)
        cand = list(range(args.min_ctx, hi))
        pos_ent = {}
        if args.select == "agreement" and len(cand) > n:
            # select positions where the J-lens and tuned-lens next-token readouts
            # AGREE (faithful next-verbalization spots -> better transfer to J-lens).
            Hc = grab2["h"][0].float()[cand]                                   # [C,d]
            jl = W_U(fnorm((J_ag @ Hc.T).T.to(W_U.weight.dtype))).float()      # [C,V]
            T = Hc + Hc @ tA.T + tb
            tl_ = W_U(fnorm(T.to(W_U.weight.dtype))).float()                   # [C,V]
            j1, t1 = jl.argmax(-1), tl_.argmax(-1)
            j5 = jl.topk(5, -1).indices; t5 = tl_.topk(5, -1).indices
            top1 = (j1 == t1)
            ov = torch.tensor([len(set(j5[c].tolist()) & set(t5[c].tolist())) / 5.0
                               for c in range(len(cand))], device=dev)
            score = torch.where(top1, 1.0 + ov, ov)                           # top1-match ranks highest
            order = torch.argsort(score, descending=True).tolist()
            pos = []
            for oi in order:                                                   # content + agreeing first
                if len(pos) >= n: break
                p = cand[oi]; ts = tok.decode([int(j1[oi])]).strip()
                if ts and ts[0].isalpha(): pos.append(p); pos_ent[p] = float(score[oi])
            for oi in order:
                if len(pos) >= n: break
                p = cand[oi]
                if p not in pos_ent: pos.append(p); pos_ent[p] = float(score[oi])
        elif args.decision_points and len(cand) > n:
            # entropy of the next-token distribution = "how many things is it
            # considering saying". Pick the highest-entropy CONTENT positions
            # (argmax token alphabetic) — the deliberation points.
            lg = logits[cand]                                  # [C, vocab]
            pr = lg.softmax(-1)
            ent = -(pr * (pr + 1e-12).log()).sum(-1)           # [C]
            order = torch.argsort(ent, descending=True).tolist()
            pos = []
            for oi in order:
                if len(pos) >= n:
                    break
                p = cand[oi]
                ts = tok.decode([int(lg[oi].argmax())]).strip()
                if ts and ts[0].isalpha():                     # content decision point
                    pos.append(p); pos_ent[p] = float(ent[oi])
            for oi in order:                                   # backfill if too few
                if len(pos) >= n:
                    break
                p = cand[oi]
                if p not in pos_ent:
                    pos.append(p); pos_ent[p] = float(ent[oi])
        else:
            sel = (torch.randperm(len(cand))[:n]).tolist()
            pos = [cand[i] for i in sel]
        # BATCHED generation: ALL positions x rollouts of this doc in ONE constant-shape generate
        # (every prefix LEFT-padded to GEN_PAD) so the fla/DeltaNet generate kernel compiles ONCE
        # globally instead of recompiling per prefix length (per-position loop was ~26h/shard).
        GEN_PAD = args.max_length
        PAD_ID = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        bids, bmask, border = [], [], []
        for p in pos:
            pref = ids[0, : p + 1]
            if pref.shape[0] > GEN_PAD:
                pref = pref[-GEN_PAD:]
            padlen = GEN_PAD - pref.shape[0]
            pad = torch.full((padlen,), PAD_ID, dtype=ids.dtype, device=dev)
            m = torch.cat([torch.zeros(padlen, dtype=torch.long, device=dev),
                           torch.ones(pref.shape[0], dtype=torch.long, device=dev)])
            for _ in range(args.rollouts):
                bids.append(torch.cat([pad, pref])); bmask.append(m); border.append(p)
        with torch.no_grad():
            g = model.generate(input_ids=torch.stack(bids), attention_mask=torch.stack(bmask),
                               min_new_tokens=args.rollout_len, max_new_tokens=args.rollout_len,
                               do_sample=True, temperature=1.0, top_p=0.95,
                               pad_token_id=tok.eos_token_id)
        newt = g[:, GEN_PAD:]
        token_ids_by_pos = {}
        for k, p in enumerate(border):
            token_ids_by_pos.setdefault(p, []).append(
                newt[k].detach().cpu().to(torch.int32).tolist())
        for p in pos:
            cont_token_rows = token_ids_by_pos[p]
            conts = [tok.decode(x, skip_special_tokens=True) for x in cont_token_rows]
            topk = torch.topk(logits[p], args.topk).indices.tolist()
            top_strs = [tok.decode([t]) for t in topk]
            for l in LAYERS:
                acts[l].append(hs[l][p].cpu().numpy().astype("float32").tolist())
            rolls.append(conts); roll_ids.append(cont_token_rows); tops.append(top_strs)
            ctxs.append(tok.decode(ids[0, max(0, p - 47): p + 1], skip_special_tokens=True))
            teacher_ids.append(ids[0, :p + 1].detach().cpu().to(torch.int32).tolist())
            continuation_ids.append(
                ids[0, p + 1:p + 1 + args.rollout_len]
                .detach().cpu().to(torch.int32).tolist())
            prompts.append(prompt_msgs); docids.append(f"d{args.doc_offset + di}")
            ents.append(pos_ent.get(p, float("nan")))
        if di % 100 == 0:
            print(f"doc {di}/{len(texts)} examples={len(rolls)}", flush=True)
        if di % 200 == 0 and di > 0 and rolls:
            print(f"[ckpt] wrote {write_out()} rows @ doc {di}", flush=True)

    n = write_out()
    print(f"wrote {args.out} ({n} rows, layers {LAYERS})")


if __name__ == "__main__":
    main()
