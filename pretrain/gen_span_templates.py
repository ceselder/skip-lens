"""Generate contexts in which a given SPAN is a natural continuation.

This is the paper's template-lens procedure (A.9.1) generalised from single words
to spans: for a target span S, ask Claude for several short passages each written
so that S is its natural continuation, ending just before S would appear, varying
topic/frame/register and never containing S.

The point of doing this per SPAN rather than per CONTEXT is to make the training
Jacobian an average over many contexts (a general disposition) while keeping the
span label exact (ground-truth verbalization). At test time the global averaged
Jacobian replaces the span-averaged one, so the only train/test difference becomes
WHICH contexts the average ran over — a far smaller shift than one context vs all.

Spans are drawn from real on-policy rollouts, so the model demonstrably produces
them; the paper used a generic word list instead. Span length defaults to 4: an
8-token span is not determined by any context (no passage makes 8 specific tokens
"natural"), while 4 is still genuinely multi-token and, per diag_span_repetition,
sampled 4-grams are diverse ordinary text rather than the numerals that dominate
the *repeating* ones.

Writes one row per generated context; the model-side pass adds P(S) so contexts
where the model would not actually say S can be dropped downstream.
"""
import argparse
import json
import os
import random
import re
import sys

import pyarrow as pq_  # noqa: F401  (kept for parity with sibling scripts)
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SYS = """You write short passages that set up a specific continuation.

You are given a TARGET string. Write {m} DIFFERENT passages, each of which ends at \
exactly the point where TARGET would come next as its natural continuation.

Hard requirements:
- Each passage must END mid-thought, precisely where TARGET would follow. Do not \
write TARGET, and do not write anything after it.
- TARGET (or any close variant of it) must NOT appear anywhere in the passage.
- Vary topic, framing, genre and register widely across the passages. Do not reuse \
the same scenario with different words.
- 25-60 words each. Plain prose, no titles, no quotes around the passage.
- A competent reader should feel TARGET is the obvious next words.

Return ONLY a JSON array of {m} strings. No commentary."""


def build_prompt(span_text, m):
    return [{"role": "user",
             "content": f"TARGET: {span_text!r}\n\nWrite the {m} passages now."}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="/workspace/data/spans_jvp/shard_5_jvp.parquet",
                    help="held out from every decoder and from the W* fit")
    ap.add_argument("--n-spans", type=int, default=150)
    ap.add_argument("--m-contexts", type=int, default=16)
    ap.add_argument("--span-len", type=int, default=4)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--base-model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/data/span_templates/prototype.jsonl")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    t = pq.read_table(args.shards, columns=["rollout_token_ids", "ctx_text",
                                            "h42_recompute_cosine", "doc_id"])
    rows = [r for r in t.slice(0, 60000).to_pylist()
            if r["h42_recompute_cosine"] >= 0.99
            and r["rollout_token_ids"] and len(r["rollout_token_ids"]) >= 16]
    random.Random(args.seed).shuffle(rows)

    # Reject degenerate targets: the repeating short spans in this corpus are
    # overwhelmingly years and round numbers, which make a useless concept set.
    def usable(txt):
        s = txt.strip()
        if len(s) < 6:
            return False
        if re.fullmatch(r"[\d\s.,:;%$/()\[\]-]+", s):
            return False
        letters = sum(c.isalpha() for c in s)
        return letters >= max(4, 0.5 * len(s))

    picked, seen = [], set()
    for r in rows:
        ids = [int(x) for x in r["rollout_token_ids"][: args.span_len]]
        txt = tok.decode(ids, skip_special_tokens=True)
        key = tuple(ids)
        if key in seen or not usable(txt):
            continue
        # must round-trip so the span is a clean token boundary
        if tok.encode(txt, add_special_tokens=False) != ids:
            continue
        seen.add(key)
        picked.append({"span_ids": ids, "span_text": txt,
                       "real_ctx_text": r["ctx_text"], "doc_id": r["doc_id"]})
        if len(picked) >= args.n_spans:
            break
    print(f"{len(picked)} usable spans (len {args.span_len}) from {len(rows)} rows",
          flush=True)
    for p in picked[:8]:
        print(f"   {p['span_text']!r}")

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        for attempt in range(4):
            try:
                msg = client.messages.create(
                    model=args.model, max_tokens=3000,
                    system=[{"type": "text",
                             "text": SYS.format(m=args.m_contexts),
                             "cache_control": {"type": "ephemeral"}}],
                    messages=build_prompt(p["span_text"], args.m_contexts))
                # content[0] can be a ThinkingBlock on Sonnet 5, so take the text
                # blocks rather than assuming position 0
                txt = "".join(b.text for b in msg.content
                              if getattr(b, "type", "") == "text").strip()
                txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
                ctxs = json.loads(txt)
                assert isinstance(ctxs, list)
                ctxs = [c for c in ctxs if isinstance(c, str)
                        and p["span_text"].strip().lower() not in c.lower()]
                if len(ctxs) >= 4:
                    return {**p, "contexts": ctxs}
            except Exception as e:
                if attempt == 3:
                    print(f"   FAIL {p['span_text']!r}: {e!r}"[:140], flush=True)
        return None

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        out = [x for x in ex.map(one, picked) if x]
    with open(args.out, "w") as fh:
        for o in out:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    n_ctx = sum(len(o["contexts"]) for o in out)
    print(f"\nwrote {args.out}: {len(out)} spans, {n_ctx} contexts "
          f"({n_ctx / max(1, len(out)):.1f} per span)", flush=True)
    if out:
        e = out[0]
        print(f"\nexample span {e['span_text']!r}")
        for c in e["contexts"][:3]:
            print(f"   …{c[-100:]}")


if __name__ == "__main__":
    main()
