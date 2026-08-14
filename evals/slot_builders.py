"""Test-time slot construction for the multi-slot workspace lens.

ONE home for the (condition -> K slot vectors) logic, imported by the eval
harness, the playground, and the tests. It previously lived duplicated in
three places, and a sign bug in the Gram-Schmidt branch shipped in two of
them before tests could see it.

All builders take h42 (the source-layer residual at one position) and the
averaged per-offset Jacobian family, and return a [K, d] slot matrix. The
injection norm-matches each slot to the local residual norm, so only slot
DIRECTIONS carry information — every builder should be read as defining
directions, not magnitudes.
"""

from __future__ import annotations

import torch

CONDITIONS = ("per_offset", "pooled_identical", "slot0_only", "no_slot0",
              "shuffled_slots", "diff", "centered", "deflated", "gs",
              "keep0_deflate_rest", "keep0_gs_rest",
              # handled by the eval loop, not here: it builds per_offset slots
              # from ANOTHER item's activation, so the readout cannot know this
              # item's context. Any score it earns is fluency, not information.
              "mismatched_context")


def sign_canonical_qr(M: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis of M's rows, sign-matched to classical Gram-Schmidt.

    ``torch.linalg.qr`` (LAPACK Householder) leaves the signs of ``diag(R)``
    unconstrained, so raw Q columns flip relative to classical Gram-Schmidt on
    roughly half of inputs — including the FIRST direction, which would come
    out anti-aligned with the first input row. Multiplying by
    ``sign(diag(R))`` restores the classical convention exactly.
    """
    q, r = torch.linalg.qr(M.T)
    s = torch.sign(torch.diagonal(r))
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (q * s).T


def build_slots(cond: str, h42: torch.Tensor, jbar, jpool, *, k: int,
                hbar: torch.Tensor | None = None, meandirs=None) -> torch.Tensor:
    """[k, d] slot matrix for `cond`.

    jbar: sequence of k averaged per-offset Jacobians (delta = 0..k-1).
    jpool: the offset-pooled averaged Jacobian.
    hbar: corpus-mean h42 (required by "centered").
    meandirs: {delta: unit mean transported direction} (required by "deflated").
    """
    per = torch.stack([jbar[d] @ h42 for d in range(k)])

    if cond == "per_offset":
        return per
    if cond == "pooled_identical":
        return (jpool @ h42).expand(k, -1).contiguous()
    if cond == "slot0_only":
        # NOTE: a zeroed slot is NOT an absent slot. karvonen injection scales
        # v/||v||, so a zero vector leaves the marker's NATURAL residual in
        # place — and training always injected all k markers, so an
        # un-injected marker is itself off-distribution. Read this condition as
        # "slot 0 + (k-1) un-injected markers", not as a 1-slot lens.
        s = torch.zeros_like(per)
        s[0] = per[0]
        return s
    if cond == "no_slot0":
        s = per.clone()
        s[0] = 0
        return s
    if cond == "shuffled_slots":
        perm = (torch.arange(k, device=per.device) + k // 2) % k
        return per[perm]
    if cond == "diff":
        # Differential slots. Geometric only: matches the training slots'
        # pairwise-cosine statistic but MIXES two horizons per slot, so it does
        # not preserve "slot d = transport to horizon d". Kept as a control.
        return torch.stack([per[0]] + [per[d] - per[d - 1] for d in range(1, k)])
    if cond == "centered":
        # Transport the mean-subtracted activation. Averaging over contexts
        # leaves a large context-INDEPENDENT component in J̄h (~79% of the norm
        # at delta=0 lies along one fixed direction), which makes the injected
        # vector nearly position-invariant. h̄ is constant, so J̄h̄ carries no
        # per-position information and removing it preserves horizon semantics.
        if hbar is None:
            raise ValueError("centered requires hbar")
        return torch.stack([jbar[d] @ (h42 - hbar) for d in range(k)])
    if cond == "deflated":
        # Same goal downstream: project each offset's mean transported
        # direction out of that slot. meandirs must be unit vectors.
        if not meandirs:
            raise ValueError("deflated requires meandirs")
        out = []
        for d in range(k):
            md = meandirs[d]
            md = md / (md.norm() + 1e-9)
            v = per[d]
            out.append(v - (v @ md) * md)
        return torch.stack(out)
    if cond == "gs":
        # Decorrelated slots (pairwise cosine 0), each rescaled to its original
        # norm. Caveat: with 0.86-collinear inputs, directions 1..k-1 are
        # noise-dominated residuals that norm-matching then amplifies to full
        # residual scale.
        return sign_canonical_qr(per)[:k] * per.norm(dim=-1, keepdim=True)
    if cond in ("keep0_deflate_rest", "keep0_gs_rest"):
        # What the full condition set implies. Evidence:
        #   per_offset  0.289  slot 0 intact + 7 near-duplicates  -> duplicates DROWN slot 0
        #   slot0_only  0.544  slot 0 intact + 7 un-injected      -> slot 0 alone carries it
        #   diff        0.586  slot 0 intact + 7 decorrelated     -> decorrelated extras are fine
        #   centered    0.218  ALL slots stripped of the shared component -> slot 0 ruined
        # So: leave slot 0 EXACTLY as J̄⁽⁰⁾h (its shared component is part of a
        # valid affine approximation to an activation — the J̄ audit found
        # h62 ≈ J̄h + b, and stripping b hurts readout), and decorrelate only
        # slots 1..k-1. Unlike "diff" this never mixes two horizons, so slot d
        # still means "the transport to horizon d".
        out = [per[0]]
        if cond == "keep0_deflate_rest":
            if not meandirs:
                raise ValueError("keep0_deflate_rest requires meandirs")
            for d in range(1, k):
                md = meandirs[d]
                md = md / (md.norm() + 1e-9)
                v = per[d]
                out.append(v - (v @ md) * md)
        else:
            rest = sign_canonical_qr(per[1:])[: k - 1] * per[1:].norm(dim=-1, keepdim=True)
            out.extend(rest[i] for i in range(k - 1))
        return torch.stack(out)
    raise ValueError(f"unknown condition {cond!r}")
