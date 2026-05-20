import torch
import pytest

from benchmarks.profile_gate0_hybrid_correction import (
    _assemble_corrected_output,
    _complement,
    _parse_heads,
)


def test_parse_heads_validates_bounds():
    assert _parse_heads("3,1,3", total_heads=5) == [1, 3]
    with pytest.raises(ValueError, match="outside"):
        _parse_heads("5", total_heads=5)


def test_complement_returns_non_selected_heads():
    assert _complement([2, 4], total_heads=6) == [0, 1, 3, 5]


def test_assemble_corrected_output_uses_sparse_only_for_trusted_heads():
    dense_exact = torch.zeros(1, 1, 4, 2)
    dense_exact[:, :, 0, :] = 10.0  # head 0
    dense_exact[:, :, 1, :] = 20.0  # head 1
    dense_exact[:, :, 2, :] = 25.0  # head 2
    dense_exact[:, :, 3, :] = 40.0  # head 4
    sparse_union = torch.zeros(1, 1, 3, 2)
    sparse_union[:, :, 0, :] = 11.0  # head 1, not trusted
    sparse_union[:, :, 1, :] = 30.0  # head 3, trusted
    sparse_union[:, :, 2, :] = 41.0  # head 4, not trusted

    actual = _assemble_corrected_output(
        total_heads=5,
        dense_exact_out=dense_exact,
        exact_heads=[0, 1, 2, 4],
        sparse_union_out=sparse_union,
        aggressive_heads=[1, 3, 4],
        trusted_heads=[3],
    )

    assert actual[:, :, 0, :].eq(10.0).all()
    assert actual[:, :, 1, :].eq(20.0).all()
    assert actual[:, :, 2, :].eq(25.0).all()
    assert actual[:, :, 3, :].eq(30.0).all()
    assert actual[:, :, 4, :].eq(40.0).all()


def test_assemble_rejects_unwritten_heads():
    with pytest.raises(ValueError, match="did not write"):
        _assemble_corrected_output(
            total_heads=3,
            dense_exact_out=torch.zeros(1, 1, 1, 2),
            exact_heads=[0],
            sparse_union_out=torch.zeros(1, 1, 1, 2),
            aggressive_heads=[1],
            trusted_heads=[1],
        )
