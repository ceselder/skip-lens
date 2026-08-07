"""Judge the fed-layer sweep (Sonnet-5). For each readout of AO(raw h_l), score TWO agreements
on the same disagreement positions, with the SAME judge structure:
  agree_jlens  = does the readout match the J-lens top-k concept(s)?   [workspace proxy]
  agree_answer = does the readout match the model's actual continuation? [the answer]
On disagreement positions these pull apart, so agree_jlens tracking high (and degrading with depth)
= faithful intermediate readout; agree_answer staying high across depths = tuned-lens decoding."""
import argparse, json, os, re, time, statistics
from concurrent.futures import ThreadPoolExecutor
import anthropic

AGREE_JLENS_SYS = ("You are given the TOP single-token predictions from a 'J-lens' readout at a position "
                   "in a language model, and a CANDIDATE multi-token readout from a different method. Score "
                   "how well the CANDIDATE agrees with / elaborates the concept(s) in the J-lens tokens "
                   "(ignore whether either is correct — judge only mutual agreement).\n"
                   "SCORE 2 = clearly the same concept(s); 1 = partial/related overlap; 0 = unrelated/contradicts.\n"
                   "Reply EXACTLY: SCORE: <0|1|2>")

AGREE_ANSWER_SYS = ("You are given several of a language model's ACTUAL sampled continuations at a position, "
                    "and a CANDIDATE multi-token readout from a different method. Score how well the CANDIDATE "
                    "agrees with / captures the concept(s) in what the model actually goes on to say (ignore "
                    "whether it is otherwise correct — judge only agreement with the actual continuation).\n"
                    "SCORE 2 = clearly the same concept(s); 1 = partial/related overlap; 0 = unrelated/contradicts.\n"
                    "Reply EXACTLY: SCORE: <0|1|2>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    cl = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def call(system, user):
        for a in range(5):
            try:
                m = cl.messages.create(model=args.model, max_tokens=256, system=system,
                                       thinking={"type": "disabled"},
                                       messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in m.content if getattr(b, "type", None) == "text").strip()
            except Exception:
                time.sleep(min(2 ** a, 20))
        return ""

    def sc(system, user):
        mm = re.search(r"SCORE:\s*([012])", call(system, user))
        return int(mm.group(1)) if mm else None

    def bul(x):
        return "\n".join(f"- {b}" for b in (x or []) if b)

    def work(r):
        cand = bul(r["readout"])
        r["agree_jlens"] = sc(AGREE_JLENS_SYS, f"J-LENS TOP TOKENS: {r['jlens_top']}\n\nCANDIDATE readout:\n{cand or '(empty)'}")
        acts = "\n".join(f"- {c}" for c in r["actual"] if c)
        r["agree_answer"] = sc(AGREE_ANSWER_SYS, f"ACTUAL CONTINUATIONS:\n{acts}\n\nCANDIDATE readout:\n{cand or '(empty)'}")
        return r

    recs = json.load(open(args.inp))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(work, recs))

    agg = {}
    for r in recs:
        a = agg.setdefault(r["fed_layer"], {"j": [], "ans": []})
        if r.get("agree_jlens") is not None: a["j"].append(r["agree_jlens"])
        if r.get("agree_answer") is not None: a["ans"].append(r["agree_answer"])

    def ms(v):
        if not v:
            return (None, None)
        h = [x / 2 for x in v]
        m = sum(h) / len(h)
        return (round(m, 4), round(statistics.pstdev(h) / len(h) ** 0.5 if len(h) > 1 else 0.0, 4))

    by_layer = {}
    for L, v in agg.items():
        e = {"n": len(v["j"])}
        e["agree_jlens"], e["agree_jlens_sem"] = ms(v["j"])
        e["agree_answer"], e["agree_answer_sem"] = ms(v["ans"])
        by_layer[str(L)] = e
    json.dump({"by_fed_layer": by_layer, "detail": recs}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print("fed-layer sweep (Sonnet-5): agreement with J-lens [workspace] vs actual continuation [answer]")
    for L in sorted(agg, reverse=True):
        s = by_layer[str(L)]
        print(f"  fed L{L:>2}: agree_JLENS={s['agree_jlens']}±{s['agree_jlens_sem']}  "
              f"agree_ANSWER={s['agree_answer']}±{s['agree_answer_sem']}  (n={s['n']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
