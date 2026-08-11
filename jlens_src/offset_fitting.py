# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fitting per-offset (token-index-resolved) Jacobian lenses.

The standard estimator (:mod:`jlens.fitting`) pools the Jacobian
``dh_target[t'] / dh_source[t]`` over ALL target positions ``t' >= t``,
producing one matrix per source layer — the collapse required to read out
with a single ``W_U``. Here we instead resolve the target offset
``delta = t' - t``, producing a FAMILY of matrices ``J^(delta)`` for
``delta in [0, n_offsets)``: the average linear map from the source-layer
residual at a position to the target-layer residual ``delta`` tokens ahead.
``J^(delta)`` is an empirical estimate of the model's horizon-``delta``
broadcast operator; the family feeds multi-slot activation-verbalizer
decoding (one transported vector per future token slot).

Estimator: one-hot cotangents are injected at COMB-SPACED target positions
(teeth), all teeth in the same backward pass. The gradient at source
position ``tooth - delta`` then contains the wanted ``J^(delta)`` row plus
contamination from later teeth at offsets ``delta + k*spacing``. Each tooth
carries a random Rademacher sign (multiplied back at extraction), so the
primary term always enters with ``+`` while contamination terms enter with
an independent random sign and average to ZERO over prompts — leakage
becomes variance, not bias.

All ``n_offsets`` estimates share the same backward passes: per-prompt cost
is identical to the pooled estimator (one forward + ``ceil(d/dim_batch)``
backwards).
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence

import torch

from jlens.fitting import SKIP_FIRST_N_POSITIONS, _atomic_save, _check_layer_indices
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)


def comb_teeth(
    seq_len: int,
    *,
    n_offsets: int,
    comb_spacing: int,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    phase: int = 0,
) -> torch.Tensor:
    """Target positions (teeth) for the comb cotangent.

    The first tooth sits at ``skip_first + n_offsets - 1 + phase`` so that
    every source position ``tooth - delta`` stays clear of the attention-sink
    region; the final position is excluded to match the pooled estimator's
    convention. ``phase`` (in ``[0, comb_spacing)``) shifts the whole comb so
    different prompts sample different absolute positions.

    Raises:
        ValueError: If the prompt is too short to place a single tooth.
    """
    if not 0 <= phase < comb_spacing:
        raise ValueError(f"phase must be in [0, {comb_spacing}), got {phase}")
    first = skip_first + n_offsets - 1 + phase
    if first >= seq_len - 1:
        raise ValueError(
            f"prompt too short for comb: seq_len={seq_len}, need > "
            f"{skip_first + n_offsets + phase}"
        )
    return torch.arange(first, seq_len - 1, comb_spacing)


def offset_jacobians_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int,
    n_offsets: int = 16,
    comb_spacing: int = 32,
    dim_batch: int = 8,
    max_seq_len: int = 512,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    phase: int = 0,
    sign_seed: int = 0,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Per-offset Jacobian estimates ``J^(delta)`` for one prompt.

    Same replicated-batch backward scheme as
    :func:`jlens.fitting.jacobian_for_prompt`, with the cotangent placed at
    comb teeth (each tooth carrying a Rademacher sign drawn from
    ``sign_seed``) instead of every valid position. After each backward, the
    rows for offset ``delta`` are read at positions ``teeth - delta``, sign-
    corrected per tooth, and averaged over teeth.

    Returns:
        ``(jacobians, seq_len, n_teeth)`` where ``jacobians[layer]`` is a
        ``[n_offsets, d_model, d_model]`` fp32 CPU tensor.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    teeth = comb_teeth(
        seq_len,
        n_offsets=n_offsets,
        comb_spacing=comb_spacing,
        skip_first=skip_first,
        phase=phase,
    )
    n_teeth = len(teeth)
    gen = torch.Generator().manual_seed(sign_seed)
    signs = torch.randint(0, 2, (n_teeth,), generator=gen).float() * 2.0 - 1.0

    jacobians = {
        layer: torch.zeros(n_offsets, d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        device = target_activation.device
        teeth_dev = teeth.to(device)
        signs_dev = signs.to(device)
        # cotangent inherits the model dtype (bf16 on GPU); index-put requires
        # matching dtypes, so pre-cast the signs used for the write.
        signs_cot = signs_dev.to(target_activation.dtype)
        batch_indices = torch.arange(dim_batch, device=device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims = min(dim_batch, d_model - dim_start)
            # Signed one-hot cotangent at dim (dim_start + b) for batch
            # element b, at every comb tooth.
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims, None],
                teeth_dev[None, :],
                dim_start + batch_indices[:n_dims, None],
            ] = signs_cot[None, :]
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                teeth_g = teeth_dev.to(grad.device, non_blocking=True)
                signs_g = signs_dev.to(grad.device, non_blocking=True)
                for delta in range(n_offsets):
                    # [n_dims, n_teeth, d]; sign-correct per tooth, then mean.
                    rows = grad[:n_dims, teeth_g - delta, :].float()
                    rows = (rows * signs_g[None, :, None]).mean(dim=1)
                    jacobians[layer][
                        delta, dim_start : dim_start + n_dims, :
                    ] = rows.cpu()
            del grads
            if pass_idx % 100 == 0 or pass_idx == n_passes - 1:
                logger.debug("    pass %d/%d", pass_idx + 1, n_passes)

    return jacobians, seq_len, n_teeth


def fit_offsets(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int],
    target_layer: int,
    n_offsets: int = 16,
    comb_spacing: int = 32,
    dim_batch: int = 8,
    max_seq_len: int = 512,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 4,
    resume: bool = True,
) -> dict[int, torch.Tensor]:
    """Fit per-offset Jacobians over a prompt corpus (running mean).

    Comb phase and Rademacher sign seed are derived deterministically from
    the prompt index, so resume reproduces the identical estimator.

    Returns:
        ``{layer: [n_offsets, d_model, d_model] fp32}`` mean Jacobians.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )
    logger.info(
        "fit_offsets: %d source layers -> L%d, %d offsets, comb=%d, %d prompts",
        len(source_layers),
        target_layer,
        n_offsets,
        comb_spacing,
        len(prompts),
    )

    meta = {
        "source_layers": source_layers,
        "target_layer": target_layer,
        "n_offsets": n_offsets,
        "comb_spacing": comb_spacing,
        "skip_first": skip_first,
        "max_seq_len": max_seq_len,
    }
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in meta.items():
            if state.get(key) != expected:
                raise ValueError(
                    f"checkpoint {checkpoint_path} has {key}={state.get(key)!r}, "
                    f"expected {expected!r}; pass resume=False to discard"
                )
        jacobian_sum, n_done, next_idx = (
            state["jacobian_sum"],
            state["n_done"],
            state["next_idx"],
        )
        logger.info("  resuming: %d/%d prompts done", next_idx, len(prompts))
    else:
        jacobian_sum = {
            layer: torch.zeros(n_offsets, d_model, d_model, dtype=torch.float32)
            for layer in source_layers
        }
        n_done, next_idx = 0, 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    **meta,
                },
                checkpoint_path,
            )

    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            per_prompt, seq_len, n_teeth = offset_jacobians_for_prompt(
                model,
                prompt,
                source_layers,
                target_layer=target_layer,
                n_offsets=n_offsets,
                comb_spacing=comb_spacing,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
                phase=prompt_idx % comb_spacing,
                sign_seed=prompt_idx,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        for layer in source_layers:
            jacobian_sum[layer] += per_prompt[layer]
        n_done += 1
        next_idx = prompt_idx + 1

        norms = [
            f"d{d}:{per_prompt[source_layers[0]][d].norm():.1f}"
            for d in range(0, n_offsets, max(1, n_offsets // 4))
        ]
        logger.info(
            "  prompt %d/%d seq=%d teeth=%d %.0fs  |J| %s",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_teeth,
            time.perf_counter() - start_time,
            " ".join(norms),
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    return {layer: jacobian_sum[layer] / n_done for layer in source_layers}
