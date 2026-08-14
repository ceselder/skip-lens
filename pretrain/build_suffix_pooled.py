"""Build SUFFIX-POOLED per-offset Jacobians (Neel's variant) from the
already-fitted exact-offset family. No refit needed.

Exact-offset (what was fitted):   J^(k) = E[ d h_tgt[t+k] / d h_src[t] ]
Suffix-pooled (this script):      S^(d) = SUM_{k >= d} J^(k)

  S^(0) is the paper's J-lens itself (it pools every t' >= t).
  S^(d) is "what this activation disposes the model to say from d tokens
  onward" — each slot keeps the average-over-everything-after property that
  makes the J-lens dispositional, and the slots differ only in how much
  near-term content is excluded.

Why this beats the exact-offset family for slot construction:
  * every slot inherits the pooling that gives the J-lens its character,
    instead of being a single-target derivative;
  * each matrix aggregates far more samples, so the deep slots stop being
    noise-limited (the exact-offset estimates at d>=8 had half-split
    cosines falling to ~0.85);
  * slot 0 is a known-good object rather than a novel one.

The sum runs over the fitted window (default 16 offsets), not to the end of
the sequence. That window captures the bulk of the pooled operator: the audit
measured cos(fresh pooled reference, sum of offsets 0..15) = 0.93, with the
missing tail carrying small norms.

Note SUM, not mean: the paper's estimator injects a cotangent at every valid
target position and averages over SOURCE positions only, so the target axis is
summed. Using a mean here would rescale each slot differently and (since the
injection norm-matches every slot anyway) only change the relative weighting
inside a slot, not between them.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jbar-dir", required=True,
                    help="dir holding Jbar_L{src}_to_L{tgt}_off{k}.npy")
    ap.add_argument("--out-dir", default=None, help="default: <jbar-dir>/suffix")
    ap.add_argument("--src-layer", type=int, default=42)
    ap.add_argument("--tgt-layer", type=int, default=62)
    ap.add_argument("--n-offsets", type=int, default=16)
    ap.add_argument("--k-slots", type=int, default=8,
                    help="how many suffix-pooled slots to write (d = 0..k-1)")
    args = ap.parse_args()
    out = args.out_dir or os.path.join(args.jbar_dir, "suffix")
    os.makedirs(out, exist_ok=True)

    stem = f"Jbar_L{args.src_layer}_to_L{args.tgt_layer}"
    J = []
    for k in range(args.n_offsets):
        p = os.path.join(args.jbar_dir, f"{stem}_off{k}.npy")
        if not os.path.exists(p):
            break
        J.append(np.load(p).astype(np.float64))
    assert len(J) >= args.k_slots, (
        f"need at least {args.k_slots} exact-offset matrices in {args.jbar_dir}, "
        f"found {len(J)}")
    print(f"loaded {len(J)} exact-offset matrices from {args.jbar_dir}", flush=True)

    # suffix sums, computed from the tail inward so each is one add
    suffix = [None] * len(J)
    acc = np.zeros_like(J[0])
    for k in range(len(J) - 1, -1, -1):
        acc = acc + J[k]
        suffix[k] = acc.copy()

    diag = {"src_layer": args.src_layer, "tgt_layer": args.tgt_layer,
            "n_offsets_used": len(J), "k_slots": args.k_slots, "slots": []}
    for d in range(args.k_slots):
        S = suffix[d].astype(np.float32)
        np.save(os.path.join(out, f"{stem}_off{d}.npy"), S)
        # cosine to the NEXT slot: nested sums overlap in every term k>d, so
        # consecutive suffix slots are far more collinear than exact offsets.
        # That is fine when BOTH train and test slots are built this way, and
        # fatal only if a decoder trained on one family is fed the other.
        nxt = suffix[d + 1] if d + 1 < len(J) else None
        cos = (float((S.astype(np.float64) * nxt).sum()
                     / (np.linalg.norm(S) * np.linalg.norm(nxt) + 1e-12))
               if nxt is not None else float("nan"))
        entry = {"delta": d, "fro_norm": float(np.linalg.norm(S)),
                 "cos_to_next_slot": cos,
                 "exact_offset_norm": float(np.linalg.norm(J[d]))}
        diag["slots"].append(entry)
        print(f"  S^{d}: |S|={entry['fro_norm']:8.2f}  "
              f"(exact J^{d} was {entry['exact_offset_norm']:7.2f})  "
              f"cos(S^{d},S^{d+1})={cos:.4f}", flush=True)

    # pooled = S^(0) by construction; write it under the pooled name too so the
    # existing eval/playground loaders find a drop-in family.
    np.save(os.path.join(out, f"{stem}_offpooled.npy"), suffix[0].astype(np.float32))
    json.dump(diag, open(os.path.join(out, "suffix_pooled_diagnostics.json"), "w"),
              indent=2)
    print(f"\nwrote {args.k_slots} suffix-pooled slots + pooled to {out}", flush=True)
    print("note: S^(0) IS the paper's J-lens (pools every t' >= t)", flush=True)


if __name__ == "__main__":
    main()
