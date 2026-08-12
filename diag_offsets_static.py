"""AUDIT (static, CPU): shipped Jbar^(delta) matrices vs shard checkpoints.

1. Re-derive the merged mean from the 3 shard checkpoints and compare
   byte-level against the shipped Jbar_L42_to_L62_off{d}.npy files.
2. Reproduce the Frobenius norms in offset_fit_diagnostics.json.
3. Check offpooled.npy == mean over the 16 per-offset matrices.
4. Basic structure stats (asymmetry of J0, diagonal mass) so the later
   transpose test is meaningful.
5. Build a held-out prompt list (pool rows NOT in the cached fitting list,
   same length filter as the fit driver) -> heldout_prompts.json for the
   GPU-side audit scripts.

Run:  /workspace/venv/bin/python diag_offsets_static.py
"""
import glob
import json
import os

import numpy as np
import torch

OUT = "/workspace/results/offset_jlens"
AUD = "/workspace/results/offset_jlens_audit"
SRC, TGT = 42, 62
SEQ = 512
N_HELDOUT = 32
os.makedirs(AUD, exist_ok=True)


def main() -> None:
    # ---- 1+2+3: merge math ------------------------------------------------
    paths = sorted(glob.glob(f"{OUT}/offset_fit_shard*of*.pt"))
    shards = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    total = sum(s["n_done"] for s in shards)
    print("shards:", [(os.path.basename(p), s["n_done"]) for p, s in zip(paths, shards)],
          "total_prompts:", total)
    for s in shards:
        print("  meta:", {k: s[k] for k in
                          ("n_offsets", "comb_spacing", "skip_first",
                           "max_seq_len", "target_layer", "source_layers")})
    acc = None
    for s in shards:
        j = s["jacobian_sum"][SRC]
        acc = j.clone() if acc is None else acc + j
    J = acc / total  # [16, d, d] fp32

    diag = json.load(open(f"{OUT}/offset_fit_diagnostics.json"))
    print(f"\n{'d':>2} {'max|npy-rederived|':>20} {'fro(npy)':>10} {'fro(json)':>10}")
    ok = True
    for d in range(J.shape[0]):
        npy = np.load(f"{OUT}/Jbar_L{SRC}_to_L{TGT}_off{d}.npy")
        red = J[d].numpy()
        max_abs = float(np.abs(npy - red).max())
        fro = float(np.linalg.norm(npy))
        fro_json = diag["per_offset"][d]["fro_norm"]
        flag = "" if max_abs < 1e-6 and abs(fro - fro_json) < 1e-3 else "  <-- MISMATCH"
        ok &= flag == ""
        print(f"{d:>2} {max_abs:>20.3e} {fro:>10.4f} {fro_json:>10.4f}{flag}")
        assert np.isfinite(npy).all(), f"non-finite values in off{d}"

    pooled = np.load(f"{OUT}/Jbar_L{SRC}_to_L{TGT}_offpooled.npy")
    dp = float(np.abs(pooled - J.mean(0).numpy()).max())
    print(f"\noffpooled max|npy - mean(off0..15)| = {dp:.3e}")
    ok &= dp < 1e-6

    # ---- 4: structure -----------------------------------------------------
    J0 = J[0].numpy()
    asym = float(np.linalg.norm(J0 - J0.T) / np.linalg.norm(J0))
    print(f"J0 asymmetry |J0-J0.T|/|J0| = {asym:.4f} (needs to be >>0 for the "
          f"transpose test to have power)")
    dmean = float(np.diag(J0).mean())
    offrms = float(np.sqrt((J0**2).mean()))
    print(f"J0 diag mean = {dmean:.5f}, overall rms entry = {offrms:.6f}")

    # cosine between consecutive offsets (should be < 1: offsets differ)
    cs = []
    for d in range(15):
        a, b = J[d].flatten(), J[d + 1].flatten()
        cs.append(float(torch.dot(a, b) / (a.norm() * b.norm())))
    print("cos(J^d, J^{d+1}) d=0..14:", " ".join(f"{c:.3f}" for c in cs))

    # ---- 5: held-out prompts ---------------------------------------------
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    print("tokenizer bos_token_id:", tok.bos_token_id,
          "has add_bos_token attr:", hasattr(tok, "add_bos_token"))
    fit = set(json.load(open("/workspace/data/ffw_fit_prompts_512tok.json")))
    pool = pq.read_table("/workspace/data/ffw_fit_pool.parquet",
                         columns=["text"]).column("text").to_pylist()
    held, n_fit_seen = [], 0
    for t in pool:
        if not t or len(t) < SEQ * 3:
            continue
        if t in fit:
            n_fit_seen += 1
            continue
        if len(tok(t, truncation=True, max_length=SEQ + 8)["input_ids"]) >= SEQ:
            held.append(t)
        if len(held) >= N_HELDOUT:
            break
    print(f"held-out prompts: {len(held)} (skipped {n_fit_seen} fit-corpus rows)")
    assert len(held) == N_HELDOUT
    with open(f"{AUD}/heldout_prompts.json", "w") as f:
        json.dump(held, f)

    print("\nSTATIC AUDIT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
