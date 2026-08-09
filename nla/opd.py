"""Loss utilities for activation-only on-policy distillation.

The teacher sees the real source-text prefix.  The student sees only the
activation-injection prompt.  Both distributions are evaluated on the same
student-sampled prefix, which keeps the training states on policy.

``reverse_kl_policy_loss`` implements the sampled reverse-KL policy-gradient
estimator used by Thinking Machines' on-policy distillation recipe.  The older
full-vocabulary forward-KL objective remains available as an explicitly named
ablation; it must not be reported as OPD.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ForwardKLCutoffLoss:
    loss: torch.Tensor
    distill_loss: torch.Tensor
    eos_loss: torch.Tensor
    kl: torch.Tensor
    distill_mask: torch.Tensor
    eos_mask: torch.Tensor

    @property
    def optimized_tokens(self) -> int:
        return int((self.distill_mask | self.eos_mask).sum().item())


@dataclass
class ReverseKLPolicyLoss:
    """Sampled ``KL(student || teacher)`` policy-gradient quantities."""

    loss: torch.Tensor
    reverse_kl_sample: torch.Tensor
    advantage: torch.Tensor
    importance_ratio: torch.Tensor
    student_sampled_logp: torch.Tensor
    teacher_sampled_logp: torch.Tensor
    valid: torch.Tensor

    @property
    def optimized_tokens(self) -> int:
        return int(self.valid.sum().item())


def reverse_kl_policy_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    sampled_tokens: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 1.0,
    behavior_logprobs: torch.Tensor | None = None,
) -> ReverseKLPolicyLoss:
    """Return the on-policy sampled reverse-KL policy-gradient loss.

    Student trajectories must have been sampled from the same policy before
    this optimization step.  ``behavior_logprobs`` may contain log-probabilities
    stored during rollout; when omitted, a detached copy of the current
    student's sampled-token log-probability is exact because no update occurs
    between rollout and this forward pass.

    With per-token (undiscounted) reverse KL, the sampled cost is
    ``log p_student(a|s) - log p_teacher(a|s)``.  Its negative is treated as a
    detached advantage and optimized with the usual importance ratio.  Merely
    differentiating the sampled log-ratio directly is not the reverse-KL
    gradient.
    """
    if teacher_logits.shape != student_logits.shape:
        raise ValueError(
            f"teacher {tuple(teacher_logits.shape)} != student "
            f"{tuple(student_logits.shape)}"
        )
    if teacher_logits.shape[:-1] != sampled_tokens.shape:
        raise ValueError(
            f"logits prefix {tuple(teacher_logits.shape[:-1])} != sampled tokens "
            f"{tuple(sampled_tokens.shape)}"
        )
    if sampled_tokens.shape != valid.shape:
        raise ValueError(
            f"sampled tokens {tuple(sampled_tokens.shape)} != valid {tuple(valid.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    valid = valid.bool()
    student_logp = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    gather_ids = sampled_tokens.unsqueeze(-1)
    current_logp = student_logp.gather(-1, gather_ids).squeeze(-1)
    teacher_sampled_logp = teacher_logp.gather(-1, gather_ids).squeeze(-1).detach()
    if behavior_logprobs is None:
        behavior_logprobs = current_logp.detach()
    elif behavior_logprobs.shape != current_logp.shape:
        raise ValueError(
            f"behavior logprobs {tuple(behavior_logprobs.shape)} != sampled logprobs "
            f"{tuple(current_logp.shape)}"
        )
    else:
        behavior_logprobs = behavior_logprobs.detach()

    reverse_kl_sample = behavior_logprobs - teacher_sampled_logp
    advantage = -reverse_kl_sample.detach()
    importance_ratio = torch.exp(current_logp - behavior_logprobs)
    if not bool(valid.any()):
        raise ValueError("reverse-KL batch has no valid sampled tokens")
    loss = -(importance_ratio[valid] * advantage[valid]).mean()
    return ReverseKLPolicyLoss(
        loss=loss,
        reverse_kl_sample=reverse_kl_sample,
        advantage=advantage,
        importance_ratio=importance_ratio,
        student_sampled_logp=current_logp,
        teacher_sampled_logp=teacher_sampled_logp,
        valid=valid,
    )


def teacher_student_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    """Return full-vocabulary KL(teacher || student) per token in float32.

    Both tensors have shape ``[..., vocab]``.  The forward direction is the
    standard distribution-distillation objective: it covers teacher-supported
    alternatives instead of only sharpening whatever the student sampled.
    """
    t_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    s_logp = F.log_softmax(student_logits.float(), dim=-1)
    return (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)


def student_teacher_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
) -> torch.Tensor:
    """Return exact full-vocabulary KL(student || teacher) per token."""
    t_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    s_logp = F.log_softmax(student_logits.float(), dim=-1)
    return (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)


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


def forward_kl_cutoff_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    valid: torch.Tensor,
    eos_token_id: int,
    kl_threshold: float | None,
    eos_weight: float = 1.0,
) -> ForwardKLCutoffLoss:
    """Compute the legacy forward-KL/EOS ablation.

    This is intentionally retained for reproducing the initial pilot, but it is
    not the sampled reverse-KL objective meant by on-policy distillation.
    """
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

    return ForwardKLCutoffLoss(
        loss=distill + eos_weight * eos,
        distill_loss=distill,
        eos_loss=eos,
        kl=kl,
        distill_mask=distill_mask,
        eos_mask=eos_mask,
    )
