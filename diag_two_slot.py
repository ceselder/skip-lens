"""Are 'current content' and 'future content' separable enough for a 2-slot lens?

The pooling sweep produced a dilemma. Weightings that include offset 0 align well
with the training target (0.37-0.40) but LOSE to raw h42, because offset 0 is
current-token content that h42 already carries. Weightings that exclude offset 0
genuinely beat raw h42 (+0.06 to +0.07) but align only ~0.15 — the same regime
where the eight-slot arms failed.

Those two facts are not in conflict if the content is separable: give the decoder
TWO slots, one per regime.

    slot 0  h42 itself            — current content, which h42 carries best
    slot 1  Σ_{d≥1} w_d J̄⁽ᵈ⁾h    — future content, which ONLY the Jacobian carries

K=2 is far from the eight-slot pathology, and unlike those slots these two have
distinct jobs, which speaks to the original failure: the eight averaged slots had
0.86 mutual collinearity against 0.29 for the local ones they were trained on.

This measures whether the design can work before anything is trained:
  * collinearity between the two slots (want LOW — redundant slots was the
    original disease);
  * each slot's own train/test alignment;
  * whether the pair spans the training target better than either alone, via the
    R² of regressing the target onto the two slots' span.
"""
import argparse
import glob
import json

import numpy as np
import pyarrow.parquet as pq
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--shards", default="/workspace/data/spans_jvp/shard_3_jvp.parquet")
ap.add_argument("--jbar-dir", default="/workspace/results/offset_jlens")
ap.add_argument("--n-rows", type=int, default=4096)
ap.add_argument("--deep-from", type=int, default=1)
ap.add_argument("--deep-to", type=int, default=16)
ap.add_argument("--out", default="/workspace/results/multislot_eval/two_slot.json")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
cos = torch.nn.functional.cosine_similarity

Jb = [torch.from_numpy(np.load(f"{args.jbar_dir}/Jbar_L42_to_L62_off{d}.npy")).float().to(dev)
      for d in range(16)]
nrm = [float(J.norm()) for J in Jb]
W = torch.zeros(16, device=dev)
for d in range(args.deep_from, args.deep_to):
    W[d] = 1.0 / nrm[d]                      # norm-equalised, the best margin found

t = pq.read_table(sorted(glob.glob(args.shards))[0],
                  columns=["activation_vector", "transported_vectors",
                           "h42_recompute_cosine"])
rows = [r for r in t.slice(0, args.n_rows * 2).to_pylist()
        if r["h42_recompute_cosine"] >= 0.99][: args.n_rows]
H = torch.tensor(np.array([r["activation_vector"] for r in rows], dtype=np.float32), device=dev)
L = torch.tensor(np.array([np.frombuffer(r["transported_vectors"], np.float16)
                           .reshape(16, -1) for r in rows], dtype=np.float32), device=dev)
N = H.shape[0]
print(f"{N} rows | deep pooling over {args.deep_from}<=d<{args.deep_to}, norm-equalised",
      flush=True)

JD = sum(W[d] * Jb[d] for d in range(16) if float(W[d]))
test0, test1 = H, (JD @ H.T).T                       # what the eval would feed
trn0 = (Jb[0] @ H.T).T * 0 + H                       # slot 0 is h42 in both phases
trn1 = (L * W.view(1, -1, 1)).sum(1)                 # the LOCAL deep pooled transport
tgt_cur = L[:, 0]                                    # current-token local transport
res = {"n_rows": N, "deep_from": args.deep_from, "deep_to": args.deep_to}

res["collinearity_test_slots"] = float(cos(test0, test1, dim=-1).abs().mean())
res["collinearity_train_slots"] = float(cos(trn0, trn1, dim=-1).abs().mean())
res["cos_slot1_train_test"] = float(cos(test1, trn1, dim=-1).mean())
res["cos_slot0_to_current_target"] = float(cos(test0, tgt_cur, dim=-1).mean())
res["cos_slot1_to_current_target"] = float(cos(test1, tgt_cur, dim=-1).mean())
res["cos_slot0_to_deep_target"] = float(cos(test0, trn1, dim=-1).mean())

print(f"\n  |cos| between the two TEST slots      {res['collinearity_test_slots']:.3f}"
      f"   <- want low; the 8 averaged slots were 0.86")
print(f"  |cos| between the two TRAIN slots     {res['collinearity_train_slots']:.3f}")
print(f"  slot 1 train/test alignment           {res['cos_slot1_train_test']:+.3f}")
print(f"  slot 0 -> current-token target        {res['cos_slot0_to_current_target']:+.3f}")
print(f"  slot 1 -> current-token target        {res['cos_slot1_to_current_target']:+.3f}"
      f"   <- should be LOW if the split is clean")
print(f"  slot 0 -> deep/future target          {res['cos_slot0_to_deep_target']:+.3f}"
      f"   <- what h42 alone knows about the future")


def captured(basis, y):
    """Per-row fraction of ‖y‖ lying in the span of the given slot vectors.

    Done per row, not by fitting across rows: with 5120-dim targets and only a
    few thousand rows a cross-row least-squares fit is underdetermined and
    reports a perfect fit for any predictor, which is meaningless.
    """
    B = torch.stack(basis, dim=1)                       # (N, k, d)
    Q, _ = torch.linalg.qr(B.transpose(1, 2))           # (N, d, k) orthonormal
    coef = torch.einsum("ndk,nd->nk", Q, y)
    return float((coef.norm(dim=-1) / (y.norm(dim=-1) + 1e-9)).mean())


for tname, tgt in (("current-token target", tgt_cur), ("deep/future target", trn1)):
    only0, only1 = captured([test0], tgt), captured([test1], tgt)
    both = captured([test0, test1], tgt)
    key = "capt_" + tname.split("/")[0].replace(" ", "_").replace("-", "_")
    res[key] = {"slot0_only": only0, "slot1_only": only1, "both": both,
                "expected_if_random": (2 / test0.shape[1]) ** 0.5}
    print(f"\n  fraction of the {tname} captured by the slots' span")
    print(f"    h42 alone                 {only0:.3f}")
    print(f"    the deep slot alone       {only1:.3f}")
    print(f"    both                      {both:.3f}"
          f"   (+{both - max(only0, only1):.3f} over the better single slot)")
    print(f"    two random directions     {res[key]['expected_if_random']:.3f}"
          f"   <- the floor")

json.dump(res, open(args.out, "w"), indent=1)
print(f"\nwrote {args.out}", flush=True)
