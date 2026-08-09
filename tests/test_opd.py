import torch

from nla.opd import first_kl_cutoff_masks, opd_loss, teacher_student_kl


def test_teacher_student_kl_is_zero_for_identical_logits():
    x = torch.tensor([[[1.0, 2.0, -1.0]]])
    assert torch.allclose(teacher_student_kl(x, x), torch.zeros(1, 1), atol=1e-7)


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
    out = opd_loss(teacher, student, valid, eos_token_id=3, kl_threshold=0.25)
    out.loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert out.optimized_tokens > 0
