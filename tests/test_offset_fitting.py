"""Ground-truth tests for per-offset Jacobian fitting (jlens_src/offset_fitting.py).

Uses a tiny causal toy model where each block mixes exactly ONE step backward,
so cross-position Jacobians have a hard structural zero: with source = block 1
and target = block 3, information travels at most 2 positions, hence
J^(delta) == 0 exactly for delta > 2. Single-tooth comb estimates are compared
against torch.autograd.functional.jacobian.
"""

import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

# jlens_src/ is the local mirror of the `jlens` package (it is symlinked to a
# dir literally named `jlens` on the GPU box). Register a package shim with
# __path__ pointing at jlens_src so `jlens.fitting` / `jlens.offset_fitting`
# resolve WITHOUT executing jlens_src/__init__.py (which needs the box-only
# _logging module).
_pkg = types.ModuleType("jlens")
_pkg.__path__ = [str(Path(__file__).resolve().parent.parent / "jlens_src")]
sys.modules.setdefault("jlens", _pkg)
_offset_fitting = importlib.import_module("jlens.offset_fitting")

comb_teeth = _offset_fitting.comb_teeth
fit_offsets = _offset_fitting.fit_offsets
offset_jacobians_for_prompt = _offset_fitting.offset_jacobians_for_prompt

D = 8
T = 40
N_BLOCKS = 4
SRC, TGT = 1, 3  # mixing spans blocks 2,3 -> J^(delta)=0 for delta > 2


class ToyBlock(nn.Module):
    """Residual block with exactly one causal mixing step: h[t] reads h[t-1]."""

    def __init__(self, seed: int):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer("W", torch.randn(D, D, generator=gen) * 0.4)
        self.register_buffer("U", torch.randn(D, D, generator=gen) * 0.4)

    def forward(self, h):
        prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        return h + torch.tanh(h @ self.W.T + prev @ self.U.T)


class ToyModel:
    """Minimal LensModel: fixed embedding table, deterministic encode."""

    def __init__(self):
        self.n_layers = N_BLOCKS
        self.d_model = D
        self.layers = nn.ModuleList(ToyBlock(seed=100 + i) for i in range(N_BLOCKS))
        gen = torch.Generator().manual_seed(7)
        self.embed = torch.randn(64, D, generator=gen)
        self.tokenizer = None

    def encode(self, text: str, *, max_length: int = T) -> torch.Tensor:
        gen = torch.Generator().manual_seed(len(text))
        n = min(max_length, T)
        return torch.randint(0, 64, (1, n), generator=gen)

    def forward(self, input_ids: torch.Tensor):
        h = self.embed[input_ids]
        for blk in self.layers:
            h = blk(h)
        return h

    def unembed(self, residual):  # pragma: no cover - unused by fitting
        return residual


def brute_force_offset_jacobians(model, input_ids, tooth, n_offsets):
    """d(out of block TGT at tooth) / d(out of block SRC at tooth-delta), exact."""
    h = model.embed[input_ids]
    for blk in model.layers[: SRC + 1]:
        h = blk(h)
    h_src = h.detach()

    def tail(x):
        y = x
        for blk in model.layers[SRC + 1 : TGT + 1]:
            y = blk(y)
        return y[0, tooth, :]

    full = torch.autograd.functional.jacobian(tail, h_src)  # [D, 1, T, D]
    return [full[:, 0, tooth - delta, :] for delta in range(n_offsets)]


class TestSingleTooth:
    """comb_spacing > seq_len -> exactly one tooth -> estimator is exact."""

    def setup_method(self):
        self.model = ToyModel()
        self.prompt = "x" * 33
        self.n_offsets = 4
        # skip_first=4 -> single tooth at position 4 + 4 - 1 = 7
        self.jac, seq_len, n_teeth = offset_jacobians_for_prompt(
            self.model,
            self.prompt,
            [SRC],
            target_layer=TGT,
            n_offsets=self.n_offsets,
            comb_spacing=1000,
            dim_batch=D,
            max_seq_len=T,
            skip_first=4,
            phase=0,
            sign_seed=3,
        )
        assert n_teeth == 1
        input_ids = self.model.encode(self.prompt, max_length=T)
        self.truth = brute_force_offset_jacobians(
            self.model, input_ids, tooth=7, n_offsets=self.n_offsets
        )

    def test_matches_autograd_ground_truth(self):
        for delta in range(3):
            est, ref = self.jac[SRC][delta], self.truth[delta]
            assert torch.allclose(est, ref, atol=1e-5), (
                f"delta={delta}: max err {(est - ref).abs().max():.2e}"
            )

    def test_structural_zero_beyond_mixing_range(self):
        # two 1-step-mixing blocks between SRC and TGT -> delta=3 unreachable
        assert self.jac[SRC][3].abs().max() < 1e-7
        assert self.truth[3].abs().max() < 1e-7

    def test_offsets_are_distinct(self):
        assert (self.jac[SRC][0] - self.jac[SRC][1]).norm() > 1e-2


class TestRademacherDecorrelation:
    def test_multi_tooth_mean_over_seeds_converges(self):
        """With several teeth, contamination is zero-mean across sign seeds:
        the seed-averaged estimate converges to the mean of per-tooth truths."""
        model = ToyModel()
        prompt = "y" * 21
        n_offsets, spacing, skip = 3, 8, 4
        input_ids = model.encode(prompt, max_length=T)
        teeth = comb_teeth(
            T, n_offsets=n_offsets, comb_spacing=spacing, skip_first=skip, phase=0
        )
        assert len(teeth) >= 3

        truth = [
            torch.stack(
                [
                    brute_force_offset_jacobians(model, input_ids, int(t), n_offsets)[d]
                    for t in teeth
                ]
            ).mean(dim=0)
            for d in range(n_offsets)
        ]

        acc = torch.zeros(n_offsets, D, D)
        n_seeds = 64
        for seed in range(n_seeds):
            jac, _, _ = offset_jacobians_for_prompt(
                model,
                prompt,
                [SRC],
                target_layer=TGT,
                n_offsets=n_offsets,
                comb_spacing=spacing,
                dim_batch=D,
                max_seq_len=T,
                skip_first=skip,
                phase=0,
                sign_seed=seed,
            )
            acc += jac[SRC]
        acc /= n_seeds

        for d in range(n_offsets):
            rel = (acc[d] - truth[d]).norm() / truth[d].norm() if truth[d].norm() > 0 \
                else (acc[d] - truth[d]).norm()
            assert rel < 0.05, f"delta={d}: rel err {rel:.3f}"


class TestFitOffsetsLoop:
    def test_checkpoint_resume_identical(self, tmp_path):
        model = ToyModel()
        prompts = ["a" * 25, "b" * 31, "c" * 27]
        kwargs = dict(
            source_layers=[SRC],
            target_layer=TGT,
            n_offsets=3,
            comb_spacing=8,
            dim_batch=D,
            max_seq_len=T,
            skip_first=4,
        )
        ckpt = str(tmp_path / "fit.pt")
        full = fit_offsets(model, prompts, checkpoint_path=ckpt, **kwargs)

        # resume from the finished checkpoint: nothing recomputed, same result
        resumed = fit_offsets(model, prompts, checkpoint_path=ckpt, **kwargs)
        assert torch.allclose(full[SRC], resumed[SRC])

        # config drift is rejected
        bad = dict(kwargs, comb_spacing=16)
        with pytest.raises(ValueError, match="comb_spacing"):
            fit_offsets(model, prompts, checkpoint_path=ckpt, **bad)
