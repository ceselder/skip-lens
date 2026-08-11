"""Coherence judge (Sonnet-5): rate each fed-layer readout's fluency/coherence, IGNORING correctness
or topic. Target-independent (a property of the lens's generation), so it runs on the raw readouts.
SCORE 2 = fluent/coherent, 1 = partial (garble/fragments/repetition), 0 = incoherent/garbage.
"""
import argparse, json, os, re, time, statistics
from concurrent.futures import ThreadPoolExecutor
import anthropic

COH_SYS = ("You are shown a short CANDIDATE text (a few sampled generations from one method). Rate ONLY "
           "its coherence and fluency as English — is it grammatical, sensible, and non-repetitive — "
           "IGNORING whether it is correct, on-topic, or meaningful.\n"
           "SCORE 2 = fluent and coherent; 1 = partially coherent (some garble, fragments, or repetition); "
           "0 = incoherent, degenerate, or repetitive garbage.\n"
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
                m = cl.messages.create(model=args.model, max_tokens=64, system=system,
                                       thinking={"type": "disabled"},
                                       messages=[{"role": "user", "content": user}])
                return "".join(b.text for b in m.content if getattr(b, "type", None) == "text").strip()
            except Exception:
                time.sleep(min(2 ** a, 20))
        return ""

    def sc(system, user):
        mm = re.search(r"SCORE:\s*([012])", call(system, user))
        return int(mm.group(1)) if mm else None

    def work(r):
        cand = "\n".join(f"- {b}" for b in (r.get("readout") or []) if b)
        r["coherence"] = sc(COH_SYS, f"CANDIDATE text:\n{cand or '(empty)'}")
        return r

    recs = json.load(open(args.inp))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        recs = list(ex.map(work, recs))

    agg = {}
    for r in recs:
        if r.get("coherence") is not None:
            agg.setdefault(r["fed_layer"], []).append(r["coherence"])

    def ms(v):
        if not v:
            return (None, None)
        h = [x / 2 for x in v]
        return (round(sum(h) / len(h), 4),
                round(statistics.pstdev(h) / len(h) ** 0.5 if len(h) > 1 else 0.0, 4))

    by = {}
    for L, v in agg.items():
        m, s = ms(v)
        by[str(L)] = {"coherence": m, "coherence_sem": s, "n": len(v)}
    json.dump({"by_fed_layer": by}, open(args.out, "w"), indent=1)
    print("coherence (Sonnet-5), 0–1:")
    for L in sorted(agg, reverse=True):
        print(f"  fed L{L:>2}: coherence={by[str(L)]['coherence']}±{by[str(L)]['coherence_sem']} (n={by[str(L)]['n']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
