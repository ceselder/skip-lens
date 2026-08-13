"""Merge per-offset Jacobian fit shards and export .npy matrices + diagnostics.

Sums shard checkpoints (running jacobian_sum / n_done), writes:
  Jbar_L{SRC}_to_L{TGT}_off{delta}.npy    [d, d] fp32, one per offset
  Jbar_L{SRC}_to_L{TGT}_offpooled.npy     mean over offsets (identical-slot control)
  offset_fit_diagnostics.json             per-delta Frobenius norm, effective rank,
                                          half-split cosine agreement (first vs
                                          second half of shards) — the convergence
                                          check deciding whether more prompts are
                                          needed and which max delta is usable.
"""
import glob
import json
import os

import numpy as np
import torch

SRC = [int(x) for x in os.environ.get("SRC", "42").split(",")]
TGT = int(os.environ.get("TGT", "62"))
OUT_DIR = os.environ.get("OUT_DIR", "/workspace/results/offset_jlens")


def load_shards() -> list[dict]:
    paths = sorted(glob.glob(f"{OUT_DIR}/offset_fit_shard*of*.pt"))
    assert paths, f"no shard checkpoints in {OUT_DIR}"
    shards = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    ref = {k: shards[0][k] for k in ("n_offsets", "comb_spacing", "target_layer")}
    for p, s in zip(paths, shards):
        for k, v in ref.items():
            assert s[k] == v, f"{p}: {k}={s[k]} != {v}"
    print(f"[merge] {len(paths)} shards, n_done: {[s['n_done'] for s in shards]}")
    return shards


def mean_of(shards: list[dict], src: int) -> torch.Tensor:
    total = sum(s["n_done"] for s in shards)
    acc = None
    for s in shards:
        j = s["jacobian_sum"][src]
        acc = j.clone() if acc is None else acc + j
    return acc / total


def _f64(x: torch.Tensor) -> torch.Tensor:
    """fp64 for reductions: fp32 accumulation over 26M elements costs ~0.2% on
    Frobenius norms and can push a cosine above 1.0 (the audit measured
    half_split_cosine = 1.0011 from this)."""
    return x.double()


def effective_rank(J: torch.Tensor, var_frac: float = 0.99) -> int:
    s = torch.linalg.svdvals(J.cuda() if torch.cuda.is_available() else J).cpu()
    cum = (s**2).cumsum(0) / (s**2).sum()
    return int((cum < var_frac).sum().item()) + 1


def main() -> None:
    shards = load_shards()
    n_offsets = shards[0]["n_offsets"]
    half = max(1, len(shards) // 2)
    diag = {"n_offsets": n_offsets,
            "total_prompts": sum(s["n_done"] for s in shards),
            "comb_spacing": shards[0]["comb_spacing"], "sources": {}}

    for src in SRC:
        J = mean_of(shards, src)                       # [n_offsets, d, d]
        J_a, J_b = mean_of(shards[:half], src), mean_of(shards[half:], src)
        rows = []
        print(f"\n=== source L{src} -> L{TGT}")
        for d in range(n_offsets):
            Jd = J[d]
            cos = torch.nn.functional.cosine_similarity(
                _f64(J_a[d]).flatten(), _f64(J_b[d]).flatten(), dim=0
            ).item() if len(shards) > 1 else float("nan")
            entry = {"delta": d, "fro_norm": _f64(Jd).norm().item(),
                     "effective_rank_99": effective_rank(Jd),
                     "half_split_cosine": cos}
            rows.append(entry)
            np.save(f"{OUT_DIR}/Jbar_L{src}_to_L{TGT}_off{d}.npy",
                    Jd.numpy().astype("float32"))
            print(f"  off{d}: |J|={entry['fro_norm']:.3f} "
                  f"rank99={entry['effective_rank_99']} split-cos={cos:.3f}")
        np.save(f"{OUT_DIR}/Jbar_L{src}_to_L{TGT}_offpooled.npy",
                J.mean(dim=0).numpy().astype("float32"))
        diag["sources"][str(src)] = rows

    json.dump(diag, open(f"{OUT_DIR}/offset_fit_diagnostics.json", "w"), indent=2)
    print(f"\n=== MERGE DONE: {OUT_DIR} ===")


if __name__ == "__main__":
    main()
