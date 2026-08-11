"""Compositional-NLA reward: leave-one-out FVE with globally-optimal composition.

The compositional NLA generates K bullets from an injected activation. Each bullet
is decoded by a FROZEN AR critic into a vector v_i ∈ R^d (the AR's single-bullet
reconstruction of the activation). We then ask: how well does the SPAN of the K
vectors reconstruct the (L2-normed) target activation g, and how much does each
individual bullet UNIQUELY contribute?

Composition — globally-optimal least-squares coefficients (not a fixed mean/sum):
    ĥ = Σ_i α*_i v_i ,   α* = argmin_α ‖g − V α‖²   (V = [v_1 … v_K] ∈ R^{d×K})
This is the best possible reconstruction from span{v_i}; α* is a K-dim lstsq, solved
in closed form per sample (batched, cheap).

Everything is measured in a WHITENED space (per-dim standardize by the dataset
mean/std of the normed target) so no single high-variance coordinate dominates FVE:
    FVE(ĥ) = 1 − ‖g − ĥ‖²_w / ‖g − μ‖²_w

Leave-one-out marginal reward for bullet i:
    r_i = FVE(all K bullets) − FVE(all bullets except i)
A redundant bullet (its vector already in the span of the others) adds ~0 → r_i≈0 →
no credit → the policy is pushed to make every bullet explain something the others
do not, while the full-set FVE drives overall reconstruction up. r_i is assigned as a
per-token advantage over bullet i's token span (see cnla docs) — the AR is a black-box
reward, never co-trained, so no gradient flows through it.

All functions are batched torch and dtype/device-agnostic. Solve in float32 for the
lstsq numerics even if activations are bf16.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------- #
# Whitening (diagonal / per-dim standardization fit once from a data sample)
# --------------------------------------------------------------------------- #
class Whitener:
    """Per-dim standardize: w(x) = (x - mu) / std. Fit on a sample of NORMED targets.

    Diagonal (not full-covariance) by design — d_model is ~5k, a full Σ^{-1/2} is a
    needless d³ eigdecomposition, and per-dim whitening is the standard 'whitened FVE'
    convention here (equalise coordinate scales, nothing more)."""

    def __init__(self, mu: torch.Tensor, std: torch.Tensor):
        self.mu = mu
        self.std = std

    @classmethod
    def fit(cls, H: torch.Tensor, eps: float = 1e-6) -> "Whitener":
        """H: [N, d] sample of L2-normed target activations."""
        H = H.float()
        mu = H.mean(0)
        std = H.std(0).clamp_min(eps)
        return cls(mu, std)

    def to(self, device) -> "Whitener":
        return Whitener(self.mu.to(device), self.std.to(device))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mu) / self.std

    def state_dict(self) -> dict:
        return {"mu": self.mu.cpu(), "std": self.std.cpu()}

    @classmethod
    def load(cls, sd: dict) -> "Whitener":
        return cls(sd["mu"], sd["std"])


# --------------------------------------------------------------------------- #
# Optimal-composition FVE
# --------------------------------------------------------------------------- #
def _lstsq_fve(V: torch.Tensor, g: torch.Tensor, ridge: float = 1e-3):
    """Best-reconstruction FVE from the span of the columns of V + the coefficients.

    V: [B, d, k] whitened bullet vectors (columns).  g: [B, d] whitened target.
    Returns (fve [B], sol [B, k, 1]).  ĥ = V α*; FVE = 1 − ‖g−ĥ‖²/‖g‖² (whitened
    target already centered, so the denominator is ‖g‖²).

    torch.linalg.lstsq on CUDA only supports driver='gels' (no rank-deficient
    driver), so solve the RIDGE-regularized normal equations (VᵀV+λI)α = Vᵀg via
    torch.linalg.solve — CUDA-native, and robust to redundant / zeroed (masked-out)
    bullet columns (a zero column just gets α≈0 from the ridge)."""
    V = torch.nan_to_num(V)
    g = torch.nan_to_num(g)
    Vt = V.transpose(1, 2)                                   # [B, k, d]
    k = Vt.shape[1]
    A = Vt @ V                                               # [B, k, k]
    # RELATIVE ridge (scales with the whitened magnitude — whitening blows vectors
    # up ~1/std, so an absolute λ is negligible vs VᵀV~thousands) + a tiny absolute
    # floor. Keeps A strictly positive-definite even for rank-deficient (two near-
    # identical bullets) or all-masked sets — else torch.linalg.solve throws "singular".
    diag_mean = A.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(0.0)   # [B]
    eye = torch.eye(k, device=A.device, dtype=A.dtype)
    A = A + (ridge * diag_mean + 1e-6).view(-1, 1, 1) * eye
    b = Vt @ g.unsqueeze(-1)                                 # [B, k, 1]
    sol = torch.linalg.solve(A, b)                           # [B, k, 1]
    ghat = (V @ sol).squeeze(-1)                             # [B, d]
    resid = ((g - ghat) ** 2).sum(-1)                        # [B]
    denom = (g ** 2).sum(-1).clamp_min(1e-12)                # [B]
    return 1.0 - resid / denom, sol


def loo_fve_rewards(
    vecs: torch.Tensor,
    target: torch.Tensor,
    whitener: Whitener,
    valid: torch.Tensor | None = None,
    threshold: float | None = None,
):
    """Leave-one-out FVE per-bullet rewards.

    Args:
      vecs:   [B, K, d] AR reconstruction of each bullet (raw activation space).
      target: [B, d]    the activation to reconstruct (will be L2-normed here).
      whitener: fit on normed targets; applied to both target and vecs.
      valid:  [B, K] bool — False marks a missing/unparsable bullet (e.g. the policy
              emitted <K bullets). Invalid bullets are dropped from every composition
              and get reward 0.
      threshold: optional ABSOLUTE per-bullet threshold on the LOO marginal.
              None (default): `r` is the raw marginal, exactly as before.
              float: `r <- clamp(r - threshold, max=0)` — only bullets whose
              marginal is BELOW the threshold carry (negative) signal, i.e.
              redundant bullets are penalized; bullets at/above the threshold get
              exactly 0 so already-good bullets are left alone (anti-KL-blowup).
              The raw marginal is always returned in `r_raw`.

    Returns dict:
      r:        [B, K] per-bullet reward consumed downstream: the marginal LOO-FVE,
                thresholded iff `threshold` is set (0 where not valid)
      r_raw:    [B, K] the raw (unthresholded) marginal — diagnostics/logging
      fve_full: [B]    FVE of the optimal composition of all valid bullets
      fve_loo:  [B, K] FVE leaving bullet i out (nan where not valid)
      coeff_full:[B, K] optimal α* for the full set (0 where not valid) — diagnostics
    """
    B, K, d = vecs.shape
    dev = vecs.device
    if valid is None:
        valid = torch.ones(B, K, dtype=torch.bool, device=dev)

    w = whitener.to(dev)
    g = torch.nn.functional.normalize(target.float(), dim=-1)   # L2-normed target
    gw = w(g)                                                   # [B, d] whitened, centered
    # normalize each AR vector to the same unit space as g (the AR trains with
    # direction-normalized MSE; keeps the lstsq well-conditioned — α still absorbs
    # any residual per-bullet scale).
    vecs = torch.nn.functional.normalize(vecs.float(), dim=-1)
    Vw = w(vecs)                                                # [B, K, d] whitened
    Vw = Vw * valid.unsqueeze(-1).float()                       # zero out invalid columns

    # full-set FVE (columns = whitened bullet vectors) + optimal coefficients
    Vcol = Vw.transpose(1, 2)                                   # [B, d, K]
    fve_full, coeff_full = _lstsq_fve(Vcol, gw)                 # [B], [B, K, 1]
    coeff_full = coeff_full.squeeze(-1) * valid.float()

    # leave-one-out: drop column i (mask to zero — a zero column contributes nothing to
    # the span, exactly equivalent to removing it from the lstsq).
    fve_loo = torch.full((B, K), float("nan"), device=dev)
    r = torch.zeros(B, K, device=dev)
    for i in range(K):
        keep = valid.clone()
        keep[:, i] = False
        Vi = (Vw * keep.unsqueeze(-1).float()).transpose(1, 2)  # [B, d, K] with col i (and invalids) zeroed
        fve_i, _ = _lstsq_fve(Vi, gw)
        fve_loo[:, i] = fve_i
        r[:, i] = fve_full - fve_i

    r = r * valid.float()
    r_raw = r
    if threshold is not None:
        # Penalize ONLY below-threshold (redundant) bullets — negative signal;
        # bullets at/above the threshold get exactly 0 (leave good bullets alone).
        # Re-apply the valid mask: for invalid bullets r_raw==0, and
        # clamp(0 - threshold, max=0) would be -threshold, not 0.
        r = torch.clamp(r - threshold, max=0.0) * valid.float()
    return {"r": r, "r_raw": r_raw, "fve_full": fve_full, "fve_loo": fve_loo,
            "coeff_full": coeff_full}


# --------------------------------------------------------------------------- #
# Self-test: redundant bullet → ~0 marginal; unique bullet → positive marginal
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    d, N = 256, 4000
    # synthetic "normed target" population to fit the whitener
    pop = torch.nn.functional.normalize(torch.randn(N, d), dim=-1)
    wh = Whitener.fit(pop)

    # one sample: target g; build 4 bullet-vectors where #4 is a near-duplicate of #1
    g = torch.nn.functional.normalize(torch.randn(1, d), dim=-1)
    a = torch.randn(1, d); b = torch.randn(1, d); c = torch.randn(1, d)
    # bullets 1,2,3 span most of g; bullet 4 ≈ bullet 1 (redundant)
    v1, v2, v3 = a, b, c
    v4 = a + 0.01 * torch.randn(1, d)          # redundant with v1
    vecs = torch.stack([v1, v2, v3, v4], dim=1)  # [1,4,d]

    out = loo_fve_rewards(vecs, g, wh)
    print("fve_full      :", round(out["fve_full"].item(), 4))
    print("per-bullet r  :", [round(x, 4) for x in out["r"][0].tolist()])
    print("  -> bullet 4 (redundant) marginal should be ≈0, << bullets 1-3")

    # sanity: a genuinely unique bullet lifts full FVE vs leaving it out
    r = out["r"][0]
    assert r[3] < r[:3].mean(), "redundant bullet should have the smallest marginal"
    print("OK: redundant bullet earns the least marginal credit")
