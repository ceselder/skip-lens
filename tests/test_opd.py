import torch

from nla.opd import (
    first_kl_cutoff_masks,
    forward_kl_cutoff_loss,
    reverse_kl_policy_loss,
    student_teacher_kl,
    teacher_student_kl,
    teacher_student_topk_tail_kl,
)
from nla.train_opd import _clip_valid_to_budget


def test_exact_token_budget_clips_only_final_valid_positions():
    valid = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=torch.bool)
    clipped = _clip_valid_to_budget(valid, remaining=4)
    assert clipped.tolist() == [
        [True, True, False, True],
        [True, False, False, False],
    ]
    assert int(clipped.sum()) == 4
    assert torch.equal(_clip_valid_to_budget(valid, 99), valid)


def test_teacher_student_kl_is_zero_for_identical_logits():
    x = torch.tensor([[[1.0, 2.0, -1.0]]])
    assert torch.allclose(teacher_student_kl(x, x), torch.zeros(1, 1), atol=1e-7)
    assert torch.allclose(student_teacher_kl(x, x), torch.zeros(1, 1), atol=1e-7)


def test_forward_and_reverse_kl_are_distinct():
    teacher = torch.tensor([[[4.0, 0.0, -2.0]]])
    student = torch.tensor([[[0.0, 0.0, 0.0]]])
    forward = teacher_student_kl(teacher, student)
    reverse = student_teacher_kl(teacher, student)
    assert forward.item() != reverse.item()
    assert forward.item() > 0
    assert reverse.item() > 0


def test_topk_tail_kl_is_zero_for_identical_logits():
    logits = torch.randn(2, 3, 11)
    kl = teacher_student_topk_tail_kl(logits, logits, top_k=4)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_topk_tail_kl_matches_binary_coarse_graining():
    teacher = torch.tensor([[[3.0, 1.0, 0.0]]])
    student = torch.tensor([[[0.0, 2.0, 1.0]]])
    got = teacher_student_topk_tail_kl(teacher, student, top_k=1)
    tp = teacher.softmax(-1)[..., 0]
    sp = student.softmax(-1)[..., 0]
    expected = tp * (tp.log() - sp.log()) + (1 - tp) * (
        (1 - tp).log() - (1 - sp).log()
    )
    assert torch.allclose(got, expected, atol=1e-6)


def test_topk_tail_kl_matches_exact_when_tail_has_one_token():
    teacher = torch.randn(2, 3, 7)
    student = torch.randn(2, 3, 7)
    exact = teacher_student_kl(teacher, student)
    coarse = teacher_student_topk_tail_kl(teacher, student, top_k=6)
    assert torch.allclose(coarse, exact, atol=2e-6)


def test_first_violation_becomes_eos_and_masks_suffix():
    kl = torch.tensor([[0.1, 0.4, 1.2, 0.2], [2.0, 0.1, 0.1, 0.1]])
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    distill, eos = first_kl_cutoff_masks(kl, valid, threshold=1.0)
    assert distill.tolist() == [[True, True, False, False], [False, False, False, False]]
    assert eos.tolist() == [[False, False, True, False], [True, False, False, False]]


def test_no_threshold_distills_every_valid_position():
    kl = torch.randn(2, 3).abs()
    valid = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    distill, eos = first_kl_cutoff_masks(kl, valid, threshold=None)
    assert torch.equal(distill, valid)
    assert not bool(eos.any())


def test_opd_loss_backpropagates_through_student_only():
    teacher = torch.randn(2, 4, 7)
    student = torch.randn(2, 4, 7, requires_grad=True)
    valid = torch.ones(2, 4, dtype=torch.bool)
    out = forward_kl_cutoff_loss(
        teacher, student, valid, eos_token_id=3, kl_threshold=0.25)
    out.loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert out.optimized_tokens > 0


def test_reverse_kl_policy_loss_is_zero_when_policies_match():
    teacher = torch.tensor([[[2.0, -1.0, 0.5]]])
    student = teacher.clone().requires_grad_(True)
    sampled = torch.tensor([[0]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    out = reverse_kl_policy_loss(teacher, student, sampled, valid)
    assert torch.allclose(out.reverse_kl_sample, torch.zeros_like(out.reverse_kl_sample))
    assert torch.allclose(out.importance_ratio, torch.ones_like(out.importance_ratio))
    assert out.loss.item() == 0.0
    out.loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student.grad))


def test_reverse_kl_policy_loss_uses_detached_advantage_and_importance_ratio():
    # The teacher favors sampled token 0 more strongly than the student, so the
    # positive advantage should increase that token's student logit.
    teacher = torch.tensor([[[3.0, 0.0]]])
    student = torch.tensor([[[0.0, 0.0]]], requires_grad=True)
    sampled = torch.tensor([[0]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    out = reverse_kl_policy_loss(teacher, student, sampled, valid)
    assert out.reverse_kl_sample.item() < 0
    assert out.advantage.item() > 0
    assert torch.allclose(out.importance_ratio, torch.ones_like(out.importance_ratio))
    out.loss.backward()
    # Gradient descent therefore raises logit 0 and lowers logit 1.
    assert student.grad[0, 0, 0] < 0
    assert student.grad[0, 0, 1] > 0


def test_reverse_kl_policy_loss_masks_padding():
    teacher = torch.randn(2, 3, 5)
    student = torch.randn(2, 3, 5, requires_grad=True)
    sampled = torch.tensor([[0, 1, 2], [3, 4, 0]])
    valid = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    out = reverse_kl_policy_loss(teacher, student, sampled, valid)
    assert out.optimized_tokens == 3
    out.loss.backward()
    assert torch.equal(student.grad[~valid], torch.zeros_like(student.grad[~valid]))
