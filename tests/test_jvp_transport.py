"""Mechanics tests for the JVP transport collector (pretrain/collect_jvp_transport.py).

Runs a tiny fp32 Llama on CPU (same HF hook pathway as qwen3_5) and checks:
  - torch.func.jvp composes with the inject/capture forward hooks,
  - JVP transports match central finite differences,
  - transports are linear in the tangent and zero for a zero tangent.

Requires the collector's device-agnostic paths, so model stays on CPU and
"cuda" placements are monkeypatched to cpu via the batch helpers directly.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pretrain import collect_jvp_transport as cjt  # noqa: E402

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

D = 32
N_LAYERS = 4


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        hidden_size=D,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def patched(monkeypatch_module):
    monkeypatch_module.setattr(cjt, "SRC_LAYER", 1)
    monkeypatch_module.setattr(cjt, "TGT_LAYER", 3)
    monkeypatch_module.setattr(cjt, "N_OFFSETS", 4)


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


def make_batch(batch_size=3, prefix_lens=(10, 13, 11)):
    torch.manual_seed(1)
    ids = torch.randint(0, 128, (batch_size, 20))
    mask = torch.ones_like(ids)
    p_pos = torch.tensor([l - 1 for l in prefix_lens[:batch_size]])
    return ids, mask, p_pos


def test_jvp_matches_finite_differences(tiny, patched):
    ids, mask, p_pos = make_batch()
    torch.manual_seed(2)
    tangents = torch.randn(ids.shape[0], D)

    t_jvp, _, _ = cjt.jvp_transports(tiny, ids, mask, p_pos, tangents)
    t_fd, _ = cjt.fd_transports(tiny, ids, mask, p_pos, tangents, eps_rel=1e-3)

    cos = torch.nn.functional.cosine_similarity(
        t_jvp.flatten(0, 1), t_fd.flatten(0, 1), dim=-1
    )
    assert cos.min() > 0.999, f"min cos {cos.min():.5f}"
    rel = (t_jvp - t_fd).norm() / t_fd.norm()
    assert rel < 1e-2, f"rel err {rel:.4f}"


def test_zero_tangent_gives_zero_transport(tiny, patched):
    ids, mask, p_pos = make_batch()
    t, _, _ = cjt.jvp_transports(tiny, ids, mask, p_pos, torch.zeros(ids.shape[0], D))
    assert t.abs().max() < 1e-6


def test_linearity_in_tangent(tiny, patched):
    ids, mask, p_pos = make_batch()
    torch.manual_seed(3)
    v = torch.randn(ids.shape[0], D)
    t1, _, _ = cjt.jvp_transports(tiny, ids, mask, p_pos, v)
    t2, _, _ = cjt.jvp_transports(tiny, ids, mask, p_pos, 2.0 * v)
    assert torch.allclose(2.0 * t1, t2, atol=1e-5)


def test_rows_do_not_interact(tiny, patched):
    """Zero tangent in row 0, nonzero in row 1: row 0's transports stay zero,
    so one shared eps scalar correctly serves independent batch rows."""
    ids, mask, p_pos = make_batch()
    torch.manual_seed(4)
    v = torch.randn(ids.shape[0], D)
    v[0] = 0
    t, _, _ = cjt.jvp_transports(tiny, ids, mask, p_pos, v)
    assert t[0].abs().max() < 1e-6
    assert t[1].abs().max() > 1e-4


def test_delta0_matches_direct_block_jacobian(tiny, patched):
    """Transport at delta=0 equals the Jacobian of blocks 2..3 at position p
    applied to the tangent (brute-force via autograd on the tail)."""
    ids, mask, p_pos = make_batch(batch_size=1, prefix_lens=(9,))
    torch.manual_seed(5)
    v = torch.randn(1, D)
    t, _, _ = cjt.jvp_transports(tiny, ids, mask, p_pos, v)

    layers = cjt.get_layers(tiny)
    captured = {}
    h = layers[1].register_forward_hook(
        lambda m, i, o: captured.__setitem__("h1", o[0] if isinstance(o, tuple) else o)
    )
    with torch.no_grad():
        tiny(input_ids=ids, attention_mask=mask, use_cache=False)
    h.remove()

    h1 = captured["h1"]
    p = int(p_pos[0])

    def tail_at_p(x_p):
        h_in = h1.clone()
        h_in[0, p] = x_p
        store = {}
        inj = layers[1].register_forward_hook(
            lambda m, i, o: (h_in, *o[1:]) if isinstance(o, tuple) else h_in
        )
        cap = layers[3].register_forward_hook(
            lambda m, i, o: store.__setitem__("h", o[0] if isinstance(o, tuple) else o)
        )
        try:
            tiny(input_ids=ids, attention_mask=mask, use_cache=False)
        finally:
            inj.remove()
            cap.remove()
        return store["h"][0, p]

    _, jvp_direct = torch.func.jvp(tail_at_p, (h1[0, p].clone(),), (v[0],))
    assert torch.allclose(t[0, 0], jvp_direct, atol=1e-4), (
        f"max err {(t[0, 0] - jvp_direct).abs().max():.2e}"
    )
