import pytest

from benchmarks.profile_seed_residual_oracle import (
    _parse_residual_sizes,
    _temporal_split,
)


def test_temporal_split_keeps_later_rows_held_out():
    train, test = _temporal_split(16, 8)
    assert train.tolist() == list(range(8))
    assert test.tolist() == list(range(8, 16))


def test_temporal_split_rejects_leakage_or_empty_holdout():
    with pytest.raises(ValueError):
        _temporal_split(8, 8)
    with pytest.raises(ValueError):
        _temporal_split(8, 0)


def test_parse_residual_sizes_is_positive_sorted_and_unique():
    assert _parse_residual_sizes("16,4,8,4") == [4, 8, 16]
    with pytest.raises(ValueError):
        _parse_residual_sizes("0,4")
