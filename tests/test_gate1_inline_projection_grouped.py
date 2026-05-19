import pytest
import torch

from benchmarks.profile_gate1_inline_projection_grouped import (
    _complement,
    _parse_ints,
    _select_heads,
    _validate_heads,
)


def test_parse_grouped_head_indices():
    assert _parse_ints("1, 3,5") == [1, 3, 5]
    assert _parse_ints("") == []


def test_grouped_head_complement_and_validation():
    assert _complement([1, 3], 5) == [0, 2, 4]
    _validate_heads([0, 2, 4], 5, name="heads")
    with pytest.raises(ValueError, match="duplicate"):
        _validate_heads([0, 0], 5, name="heads")
    with pytest.raises(ValueError, match="outside"):
        _validate_heads([5], 5, name="heads")


def test_select_heads_preserves_requested_order():
    tensor = torch.arange(1 * 2 * 4 * 3).reshape(1, 2, 4, 3)
    selected = _select_heads(tensor, [3, 1])
    assert selected.shape == (1, 2, 2, 3)
    torch.testing.assert_close(selected[:, :, 0], tensor[:, :, 3])
    torch.testing.assert_close(selected[:, :, 1], tensor[:, :, 1])
