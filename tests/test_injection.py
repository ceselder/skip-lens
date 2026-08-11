"""Unit tests for K-marker injection (nla/injection.py).

The K=1 cases pin the pre-existing single-marker behavior; the K>1 cases
cover the multi-slot extension (<concept>㊗㊗…㊗</concept>) used by the
per-offset transported-span AV.
"""

import pytest
import torch

from nla.injection import (
    _valid_marker_positions,
    inject_at_marked_positions,
    karvonen_inject_in_residual,
    marker_well_formed,
)

INJ, L, R = 900, 800, 801  # marker + canonical <concept> / </concept> neighbors
D = 4


def ids(*toks):
    return torch.tensor([list(toks)], dtype=torch.long)


def vecs(n, scale=1.0):
    return torch.arange(1, n + 1, dtype=torch.float32)[:, None].repeat(1, D) * scale


class TestValidMarkerPositions:
    def test_single_marker_canonical(self):
        assert _valid_marker_positions(ids(1, L, INJ, R, 2)[0], INJ, L, R) == [2]

    def test_single_marker_wrong_neighbors(self):
        assert _valid_marker_positions(ids(1, 5, INJ, 6, 2)[0], INJ, L, R) == []

    def test_run_of_k(self):
        row = ids(1, L, INJ, INJ, INJ, R, 2)[0]
        assert _valid_marker_positions(row, INJ, L, R) == [2, 3, 4]

    def test_echoed_run_rejected_whole(self):
        # marker run inside response text (no canonical flanks): under the old
        # per-position check the interior of a 3-run would pass; runs must not.
        row = ids(1, 5, INJ, INJ, INJ, 6, 2)[0]
        assert _valid_marker_positions(row, INJ, L, R) == []

    def test_run_at_boundary_rejected(self):
        assert _valid_marker_positions(ids(INJ, INJ, R)[0], INJ, L, R) == []
        assert _valid_marker_positions(ids(L, INJ, INJ)[0], INJ, L, R) == []

    def test_two_valid_runs_both_accepted_in_order(self):
        row = ids(L, INJ, R, 5, L, INJ, INJ, R)[0]
        assert _valid_marker_positions(row, INJ, L, R) == [1, 5, 6]


class TestInjectAtMarkedPositions:
    def test_single_marker_overwrites(self):
        input_ids = ids(1, L, INJ, R, 2)
        emb = torch.zeros(1, 5, D)
        out = inject_at_marked_positions(input_ids, emb, vecs(1), INJ, L, R)
        assert torch.equal(out[0, 2], vecs(1)[0])
        assert out[0, [0, 1, 3, 4]].abs().sum() == 0

    def test_k_markers_slot_order(self):
        k = 3
        input_ids = ids(1, L, INJ, INJ, INJ, R, 2)
        emb = torch.zeros(1, 7, D)
        out = inject_at_marked_positions(input_ids, emb, vecs(k), INJ, L, R)
        for slot in range(k):
            assert torch.equal(out[0, 2 + slot], vecs(k)[slot]), f"slot {slot}"

    def test_count_mismatch_raises(self):
        input_ids = ids(1, L, INJ, INJ, R, 2)
        emb = torch.zeros(1, 6, D)
        with pytest.raises(RuntimeError, match="injection sites"):
            inject_at_marked_positions(input_ids, emb, vecs(1), INJ, L, R)

    def test_echoed_run_not_counted(self):
        # valid 2-run in prompt + echoed 2-run without flanks in "response"
        input_ids = ids(L, INJ, INJ, R, 7, INJ, INJ, 8)
        emb = torch.zeros(1, 8, D)
        out = inject_at_marked_positions(input_ids, emb, vecs(2), INJ, L, R)
        assert torch.equal(out[0, 1], vecs(2)[0])
        assert torch.equal(out[0, 2], vecs(2)[1])
        assert out[0, [5, 6]].abs().sum() == 0

    def test_seq_slice_global_indexing(self):
        # K=2 run at global positions 2,3; embeddings shard covers [3, 6) only.
        input_ids = ids(1, L, INJ, INJ, R, 2)
        emb_shard = torch.zeros(1, 3, D)
        out = inject_at_marked_positions(
            input_ids, emb_shard, vecs(2), INJ, L, R, seq_slice=(3, 6)
        )
        # global pos 3 → local row 0 gets vector index 1; pos 2 outside shard
        assert torch.equal(out[0, 0], vecs(2)[1])
        assert out[0, [1, 2]].abs().sum() == 0


class TestKarvonenInject:
    def test_norm_matched_add_per_slot(self):
        k = 2
        input_ids = ids(1, L, INJ, INJ, R, 2)
        resid = torch.randn(1, 6, D)
        base = resid.clone()
        v = torch.randn(k, D)
        out = karvonen_inject_in_residual(input_ids, resid, v, INJ, L, R)
        for slot, p in enumerate([2, 3]):
            expected = base[0, p] + base[0, p].norm() * v[slot] / (v[slot].norm() + 1e-9)
            assert torch.allclose(out[0, p], expected, atol=1e-5), f"slot {slot}"
        untouched = [0, 1, 4, 5]
        assert torch.equal(out[0, untouched], base[0, untouched])

    def test_count_mismatch_raises(self):
        input_ids = ids(1, L, INJ, INJ, INJ, R)
        resid = torch.randn(1, 6, D)
        with pytest.raises(RuntimeError, match="marker sites"):
            karvonen_inject_in_residual(input_ids, resid, vecs(2), INJ, L, R)


class TestMarkerWellFormed:
    def test_default_single_marker(self):
        assert marker_well_formed([1, L, INJ, R, 2], INJ, L, R)
        assert not marker_well_formed([1, 5, INJ, 6, 2], INJ, L, R)
        assert not marker_well_formed([L, INJ, R, L, INJ, R], INJ, L, R)

    def test_k_markers(self):
        good = [1, L, INJ, INJ, INJ, R, 2]
        assert marker_well_formed(good, INJ, L, R, n_markers=3)
        assert not marker_well_formed(good, INJ, L, R, n_markers=2)
        assert not marker_well_formed(good, INJ, L, R)  # default expects 1

    def test_split_runs_rejected(self):
        # right total count but two separate runs → not one contiguous slot block
        split = [L, INJ, R, 5, L, INJ, INJ, R]
        assert not marker_well_formed(split, INJ, L, R, n_markers=3)

    def test_stray_marker_rejected(self):
        stray = [1, L, INJ, INJ, R, 7, INJ, 8]
        assert not marker_well_formed(stray, INJ, L, R, n_markers=2)
