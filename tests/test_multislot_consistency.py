"""Adversarial-review tests for the multi-slot transported-lens changeset.

Covers the cross-file invariants no single-module test pins down:

  1. SLOT ORDERING: the training reshape ([B, K*d] slot-major -> view(B*K, d)
     in nla.train_sft._av_prepare_chunk), the eval/playground tiling
     ([K, d].repeat(B, 1) in evals/multislot_fed_eval.brollout_slots and
     interface_multislot._roll), and the consumption order of
     karvonen_inject_in_residual (batch rows outer, marker positions inner)
     must all agree.  A transposed / interleaved layout anywhere would keep
     training CE healthy (the net can learn ANY fixed permutation) while
     silently scrambling the train->eval correspondence.
  2. prepare_batch / batch_rows alignment in pretrain.collect_jvp_transport
     (length-sorting must never desynchronize tangents from labels).
  3. comb_teeth source positions never entering the attention-sink region.
  4. Rademacher tooth signs applied exactly twice (write + extract): the
     single-tooth estimate is exact for BOTH tooth signs.
  5. Zero-vector Karvonen semantics used by the slot0_only / no_slot0
     knockouts: a zero slot leaves the marker residual untouched.
  6. The `gs` condition in evals/multislot_fed_eval.slots_for /
     diag_slot_collinearity: torch.linalg.qr does NOT fix column signs, so
     the "Gram-Schmidt" slots come out direction-flipped on data-dependent
     items (slot 0 flips whenever the leading pivot is positive).  The
     strict-xfail test documents the bug; the sibling test pins the fix
     (multiply q's columns by sign(diag(R))).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

from nla.injection import karvonen_inject_in_residual  # noqa: E402
from slot_builders import build_slots, sign_canonical_qr  # noqa: E402
from nla.train_sft import _av_prepare_chunk  # noqa: E402
from pretrain.collect_jvp_transport import batch_rows, prepare_batch  # noqa: E402

# jlens shim identical to test_offset_fitting.py (import that module so the
# shim + ToyModel are shared and sys.modules stays consistent).
import test_offset_fitting as tof  # noqa: E402

K = 3       # slots per row
B = 3       # batch rows
D = 16      # model dim (>= B*K so each slot can be a distinct one-hot)

INJ_CHAR = "㊗"  # ㊗
INJ = ord(INJ_CHAR)
LEFT, RIGHT = ord("L"), ord("R")
INJECT_PLACEHOLDER = "<INJECT>"


class CharTok:
    """Character-level fake tokenizer, just enough for _av_prepare_chunk."""

    eos_token = "%"
    eos_token_id = ord("%")

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        assert len(msgs) == 1 and msgs[0]["role"] == "user"
        return "[" + msgs[0]["content"] + "]"

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]


def slot_vec(b, k):
    """Distinct unit direction for (row b, slot k): one-hot at dim b*K + k."""
    v = torch.zeros(D)
    v[b * K + k] = 1.0
    return v


def make_rows():
    """B AV rows in the finalize_jvp_spans convention: activation_vector is
    the [K, d] slot matrix flattened SLOT-MAJOR (slot 0's d floats first)."""
    rows = []
    for b in range(B):
        slots = torch.stack([slot_vec(b, k) for k in range(K)])  # [K, d]
        rows.append({
            "prompt": [{"role": "user",
                        "content": "L" + INJECT_PLACEHOLDER * K + "R"}],
            "response": "ab",
            "activation_vector": slots.reshape(-1).tolist(),
        })
    return rows


def injected_directions(ids, out, base_resid):
    """For every marker position, recover which one-hot direction was added."""
    dirs = []
    for b in range(ids.shape[0]):
        for p in (ids[b] == INJ).nonzero().flatten().tolist():
            delta = out[b, p] - base_resid[b, p]
            if delta.abs().max() < 1e-6:
                dirs.append(None)  # untouched marker
            else:
                dirs.append(int(delta.argmax()))
    return dirs


class TestSlotOrderTrainEvalAgree:
    """Item 1: the three layouts consume identically."""

    def test_train_reshape_is_slot_major_row_major(self):
        rows = make_rows()
        ids, attn, mask, v_batch = _av_prepare_chunk(
            rows, CharTok(), INJ_CHAR, "cpu", n_slots=K)
        assert v_batch.shape == (B * K, D)
        for b in range(B):
            for k in range(K):
                assert torch.equal(v_batch[b * K + k], slot_vec(b, k)), (
                    f"v_batch row {b * K + k} is not (row {b}, slot {k}): "
                    f"the [B, K*d] -> [B*K, d] reshape is not row-major slot-major"
                )

    def test_training_injection_places_slot_k_on_marker_k(self):
        rows = make_rows()
        ids, attn, mask, v_batch = _av_prepare_chunk(
            rows, CharTok(), INJ_CHAR, "cpu", n_slots=K)
        resid = torch.ones(ids.shape[0], ids.shape[1], D) * 2.0
        out = karvonen_inject_in_residual(ids, resid, v_batch, INJ, LEFT, RIGHT)
        dirs = injected_directions(ids, out, resid)
        # batch row b's k-th marker must carry (row b, slot k) = one-hot b*K+k
        assert dirs == list(range(B * K)), (
            f"marker->vector assignment {dirs} != row-major expectation: "
            f"training injection order disagrees with the v_batch reshape"
        )

    def test_eval_repeat_layout_matches_training_layout(self):
        """brollout_slots does slots[K,d].repeat(B,1); every generated batch
        row must see slots 0..K-1 in order — identical to a training batch
        whose rows all carry the same slot matrix."""
        slots = torch.stack([slot_vec(0, k) for k in range(K)])  # [K, d]
        prompt = [LEFT] + [INJ] * K + [RIGHT]
        ids = torch.tensor([prompt] * B)
        vec_eval = slots.repeat(B, 1)  # the eval/playground layout
        # equivalent training layout: each row stores slots.flatten()
        vec_train = torch.tensor(
            np.stack([slots.reshape(-1).numpy()] * B)).view(B * K, D)
        assert torch.equal(vec_eval, vec_train)
        resid = torch.ones(B, len(prompt), D)
        out = karvonen_inject_in_residual(ids, resid, vec_eval, INJ, LEFT, RIGHT)
        dirs = injected_directions(ids, out, resid)
        assert dirs == [k for _ in range(B) for k in range(K)], (
            "eval tiling does not deliver slots 0..K-1 per batch row"
        )

    def test_transposed_layout_would_be_caught(self):
        """Sensitivity check: a dim-major (transposed) flatten — the layout the
        code would produce if finalize wrote tv.T — must NOT pass the same
        assertion, i.e. these tests can actually fail."""
        rows = make_rows()
        for r in rows:
            S = torch.tensor(r["activation_vector"]).view(K, D)
            r["activation_vector"] = S.T.reshape(-1).tolist()  # corrupt: dim-major
        ids, attn, mask, v_batch = _av_prepare_chunk(
            rows, CharTok(), INJ_CHAR, "cpu", n_slots=K)
        match = all(
            torch.equal(v_batch[b * K + k], slot_vec(b, k))
            for b in range(B) for k in range(K)
        )
        assert not match, "test cannot discriminate layouts — rewrite it"


class TestPrepareBatchAlignment:
    """Item 2: length-sorting happens ONLY in batch_rows; prepare_batch must
    tensorize in the given order so zip(batch, tangents/outputs) stays aligned."""

    @staticmethod
    def _rows(n=10):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(n):
            plen = int(rng.integers(5, 30))
            rows.append({
                "doc_id": f"d{i}",
                "teacher_input_ids": [1000 + i] * (plen - 1) + [2000 + i],
                "rollout_token_ids": [[3000 + i] + [i] * 15],  # list-of-lists
            })
        return rows

    def test_batches_are_partition_and_rows_align(self):
        rows = self._rows()
        seen = []
        for batch in batch_rows(rows, 4):
            ids, mask, p_pos = prepare_batch(batch, pad_id=0, device="cpu")
            for i, r in enumerate(batch):
                plen = len(r["teacher_input_ids"])
                assert int(p_pos[i]) == plen - 1
                # position p holds THIS row's last prefix token: tangents and
                # output labels zipped by index land on the right sequence.
                assert int(ids[i, p_pos[i]]) == r["teacher_input_ids"][-1]
                assert int(ids[i, plen]) == r["rollout_token_ids"][0][0]
                assert int(mask[i].sum()) == plen + len(r["rollout_token_ids"][0])
                seen.append(r["doc_id"])
        assert sorted(seen) == sorted(r["doc_id"] for r in self._rows()), (
            "batch_rows dropped or duplicated rows"
        )

    def test_probe_permutation_is_a_derangement(self):
        for n in range(2, 9):
            perm = (torch.arange(n) + 1) % n  # the probe path's permutation
            assert (perm != torch.arange(n)).all()
            assert sorted(perm.tolist()) == list(range(n))


class TestCombTeeth:
    """Item 3: source positions teeth-delta never dip into the sink region."""

    def test_sources_stay_at_or_above_skip_first(self):
        comb_teeth = tof._offset_fitting.comb_teeth
        for seq_len in (64, 100, 257, 512):
            for n_offsets in (1, 8, 16):
                for spacing in (17, 32):
                    for phase in (0, 3, spacing - 1):
                        for skip in (0, 4, 16):
                            try:
                                teeth = comb_teeth(
                                    seq_len, n_offsets=n_offsets,
                                    comb_spacing=spacing, skip_first=skip,
                                    phase=phase)
                            except ValueError:
                                continue  # too short: loud skip, fine
                            assert len(teeth) > 0
                            assert int(teeth.min()) - (n_offsets - 1) >= skip
                            assert int(teeth.max()) < seq_len - 1


class TestRademacherSignRoundTrip:
    """Item 4: sign applied at cotangent write AND extraction — the single
    tooth estimate must be exact whichever sign the tooth drew."""

    def test_single_tooth_exact_for_both_tooth_signs(self):
        model = tof.ToyModel()
        prompt = "z" * 29
        n_offsets = 3
        input_ids = model.encode(prompt, max_length=tof.T)
        truth = tof.brute_force_offset_jacobians(
            model, input_ids, tooth=6, n_offsets=n_offsets)
        signs_seen = set()
        for seed in range(6):
            gen = torch.Generator().manual_seed(seed)
            sign = float(torch.randint(0, 2, (1,), generator=gen)) * 2.0 - 1.0
            signs_seen.add(sign)
            jac, _, n_teeth = tof.offset_jacobians_for_prompt(
                model, prompt, [tof.SRC], target_layer=tof.TGT,
                n_offsets=n_offsets, comb_spacing=1000, dim_batch=tof.D,
                max_seq_len=tof.T, skip_first=4, phase=0, sign_seed=seed)
            assert n_teeth == 1
            for d in range(n_offsets):
                assert torch.allclose(jac[tof.SRC][d], truth[d], atol=1e-5), (
                    f"seed={seed} (tooth sign {sign:+.0f}) delta={d}: sign not "
                    f"round-tripped (double- or missed application)"
                )
        assert signs_seen == {1.0, -1.0}, "seeds 0..5 must exercise both signs"


class TestZeroSlotSemantics:
    """Item 5: slot0_only / no_slot0 zero some slots; the Karvonen hook then
    adds ||h||*0 — the marker keeps its NATURAL residual (marker-token
    embedding through the first blocks), it is not removed or blanked."""

    def test_zero_vector_leaves_marker_untouched_but_is_consumed(self):
        prompt = [LEFT] + [INJ] * K + [RIGHT]
        ids = torch.tensor([prompt])
        resid = torch.randn(1, len(prompt), D)
        vecs = torch.stack([slot_vec(0, 0),
                            torch.zeros(D),
                            slot_vec(0, 2)])
        out = karvonen_inject_in_residual(ids, resid, vecs, INJ, LEFT, RIGHT)
        p0, p1, p2 = 1, 2, 3
        assert not torch.equal(out[0, p0], resid[0, p0])   # injected
        assert torch.equal(out[0, p1], resid[0, p1])       # zero slot: no-op
        assert not torch.equal(out[0, p2], resid[0, p2])   # injected
        # and the count check consumed all three (no RuntimeError raised)


def _near_collinear_slots(seed, k=8, d=64):
    """Slot matrices statistically like the averaged transports (pairwise
    cos ~0.9): one shared direction plus small per-slot noise."""
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(d, generator=gen)
    return torch.stack([
        base * (1.0 + 0.05 * i) + 0.1 * torch.randn(d, generator=gen)
        for i in range(k)
    ])


def _classical_gram_schmidt(P):
    out = []
    for v in P:
        u = v.clone()
        for w in out:
            u = u - (u @ w) * w
        out.append(u / u.norm())
    return torch.stack(out)


class TestGsConditionQrSigns:
    """Item 6: the `gs` condition (evals/multislot_fed_eval.py slots_for and
    diag_slot_collinearity.py) computes q, _ = qr(per.T); q.T[:K] * norms.
    torch.linalg.qr's Householder QR leaves diag(R) signs unconstrained, so
    each "Gram-Schmidt" slot is the classical direction times an arbitrary
    data-dependent sign — including slot 0, which anti-aligns with J̄⁽⁰⁾h
    whenever the leading pivot is positive (~half the eval items)."""

    def test_real_gs_builder_preserves_slot_directions(self):
        """Guards the SHIPPED builder (evals/slot_builders.py), not a copy:
        the sign bug existed in two duplicated copies before extraction."""
        flipped = []
        for seed in range(40):
            per = _near_collinear_slots(seed)
            k, d = per.shape
            eye = torch.eye(d)
            jbar = {i: torch.outer(per[i], torch.ones(d)) / d for i in range(k)}
            # build_slots computes jbar[i] @ h42; use h42 = 1-vector so that
            # jbar[i] @ h42 == per[i] exactly, exercising the real code path.
            h42 = torch.ones(d)
            gs = build_slots("gs", h42, jbar, eye, k=k)
            ref = _classical_gram_schmidt(per)
            cos = F.cosine_similarity(gs, ref * per.norm(dim=-1, keepdim=True),
                                      dim=-1)
            if (cos < 0).any():
                flipped.append((seed, cos.tolist()))
        assert not flipped, (
            f"CONFIRMED BUG: {len(flipped)}/40 seeded slot matrices give "
            f"sign-flipped 'Gram-Schmidt' slots (first: seed "
            f"{flipped[0][0]}, per-slot cos {flipped[0][1]}). "
            f"torch.linalg.qr does not fix diag(R) >= 0."
        )

    def test_sign_corrected_qr_matches_classical_gram_schmidt(self):
        """The smallest fix: q *= sign(diag(R)). Exactly classical GS then."""
        for seed in range(40):
            per = _near_collinear_slots(seed)
            q, r = torch.linalg.qr(per.T)
            s = torch.sign(torch.diagonal(r))
            s[s == 0] = 1.0
            q = q * s  # flip columns so diag(R) >= 0
            gs = q.T[: per.shape[0]] * per.norm(dim=-1, keepdim=True)
            ref = _classical_gram_schmidt(per) * per.norm(dim=-1, keepdim=True)
            assert torch.allclose(gs, ref, atol=1e-4), f"seed {seed}"

    def test_slot0_flip_condition_is_the_leading_pivot_sign(self):
        """Pin the mechanism: LAPACK Householder makes R00 = -sign(x0)*||x||,
        so slot 0 flips exactly when per[0]'s first coordinate is positive."""
        for lead in (1.0, -1.0):
            per = _near_collinear_slots(11)
            per[0, 0] = lead * (per[0, 0].abs() + 1.0)
            q, r = torch.linalg.qr(per.T)
            gs0 = q.T[0]
            cos0 = float(F.cosine_similarity(gs0, per[0], dim=0))
            expected_sign = -lead  # anti-aligned iff leading entry positive
            assert cos0 * expected_sign > 0.9, (
                f"lead={lead:+.0f}: cos(gs[0], per[0])={cos0:+.3f} — "
                f"QR sign convention changed; re-verify slots_for('gs')"
            )
