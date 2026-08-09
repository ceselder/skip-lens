"""Loss utilities for activation-only on-policy distillation.

The teacher sees the real source-text prefix.  The student sees only the
activation-injection prompt.  Both distributions are evaluated on the same
student-sampled prefix, which keeps the training states on policy.

When the teacher-to-student KL first exceeds a configured cutoff, the student
is trained to emit EOS at that position and the rest of that rollout is
ignored.  The cutoff is intentionally a *routing rule*, not a clipped loss:
forcing a continuation after the activation has stopped being informative is
exactly the hallucination mode this experiment is intended to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class OPDLoss:
    loss: torch.Tensor
    distill_loss: torch.Tensor
    eos_loss: torch.Tensor
    kl: torch.Tensor
    distill_mask: torch.Tensor
    eos_mask: torch.Tensor

    @property
    def optimized_tokens(self) -> int:
        return int((self.distill_mask | self.eos_mask).sum().item())


def teacher_student_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    """Return KL(teacher || student) per token in float32.

    Both tensors have shape ``[..., vocab]``.  The forward direction is the
    standard distribution-distillation objective: it covers teacher-supported
    alternatives instead of only sharpening whatever the student sampled.
    """
    t_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    s_logp = F.log_softmax(student_logits.float(), dim=-1)
    return (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)


def first_kl_cutoff_masks(
    kl: torch.Tensor,
    valid: torch.Tensor,
    threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build masks for distillation positions and the first EOS-abstention.

    ``valid`` marks real rollout positions.  With no threshold all valid
    positions are distilled.  Otherwise, positions strictly before the first
    KL violation are distilled; the violating position receives an EOS target;
    everything after it is ignored.
    """
    if kl.shape != valid.shape:
        raise ValueError(f"kl {tuple(kl.shape)} != valid {tuple(valid.shape)}")
    valid = valid.bool()
    if threshold is None:
        return valid, torch.zeros_like(valid)

    bad = (kl.detach() > threshold) & valid
    # cumsum includes the first bad position.  Distill only while it is zero.
    bad_seen = bad.to(torch.int64).cumsum(dim=-1)
    distill = valid & (bad_seen == 0)
    eos = bad & (bad_seen == 1)
    return distill, eos


def opd_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    valid: torch.Tensor,
    eos_token_id: int,
    kl_threshold: float | None,
    eos_weight: float = 1.0,
) -> OPDLoss:
    """Compute masked full-distribution KL plus first-violation EOS CE."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"teacher {tuple(teacher_logits.shape)} != student "
            f"{tuple(student_logits.shape)}"
        )
    kl = teacher_student_kl(teacher_logits, student_logits)
    distill_mask, eos_mask = first_kl_cutoff_masks(kl, valid, kl_threshold)

    zero = student_logits.float().sum() * 0.0
    if bool(distill_mask.any()):
        distill = kl[distill_mask].mean()
    else:
        distill = zero

    if bool(eos_mask.any()):
        eos_targets = torch.full(
            (int(eos_mask.sum().item()),),
            eos_token_id,
            dtype=torch.long,
            device=student_logits.device,
        )
        eos = F.cross_entropy(student_logits.float()[eos_mask], eos_targets)
    else:
        eos = zero

    return OPDLoss(
        loss=distill + eos_weight * eos,
        distill_loss=distill,
        eos_loss=eos,
        kl=kl,
        distill_mask=distill_mask,
        eos_mask=eos_mask,
    )
