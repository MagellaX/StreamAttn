import torch

from benchmarks.profile_qwen_routed_projection_floor import (
    packed_qkv_params,
    parse_batch_sizes,
    parse_layers,
)


class _DummyAttention(torch.nn.Module):
    def __init__(self, *, bias: bool):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 6, bias=bias)
        self.k_proj = torch.nn.Linear(4, 2, bias=bias)
        self.v_proj = torch.nn.Linear(4, 2, bias=bias)
        self.o_proj = torch.nn.Linear(6, 4, bias=bias)


def test_parse_layers_preserves_order_and_deduplicates():
    assert parse_layers("0, 14;16,14, 35") == [0, 14, 16, 35]


def test_parse_batch_sizes_preserves_order_and_deduplicates():
    assert parse_batch_sizes("4,8;4,16") == [4, 8, 16]


def test_packed_qkv_params_concatenates_weight_and_bias():
    module = _DummyAttention(bias=True)
    weight, bias, sizes = packed_qkv_params(module)

    assert sizes == (6, 2, 2)
    assert weight.shape == (10, 4)
    assert bias is not None
    assert bias.shape == (10,)

    hidden = torch.randn(3, 1, 4)
    packed = torch.nn.functional.linear(hidden, weight, bias)
    q, k, v = torch.split(packed, sizes, dim=-1)

    torch.testing.assert_close(q, module.q_proj(hidden))
    torch.testing.assert_close(k, module.k_proj(hidden))
    torch.testing.assert_close(v, module.v_proj(hidden))


def test_packed_qkv_params_handles_missing_bias():
    module = _DummyAttention(bias=False)
    weight, bias, sizes = packed_qkv_params(module)

    assert sizes == (6, 2, 2)
    assert weight.shape == (10, 4)
    assert bias is None
