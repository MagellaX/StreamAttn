import pytest
import torch
import torch.nn.functional as F

from stream_attention.kernels.qwen_o_proj_triton import (
    TRITON_AVAILABLE,
    qwen_o_proj_triton_forward,
)


def test_qwen_o_proj_triton_requires_cuda_tensors():
    x = torch.randn(2, 1, 8, dtype=torch.float16)
    weight = torch.randn(8, 8, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="Triton|CUDA"):
        qwen_o_proj_triton_forward(x, weight)


@pytest.mark.skipif(not torch.cuda.is_available() or not TRITON_AVAILABLE, reason="CUDA/Triton required")
@pytest.mark.parametrize("has_bias", [False, True])
def test_qwen_o_proj_triton_matches_f_linear(has_bias):
    torch.manual_seed(123)
    x = torch.randn(5, 1, 64, device="cuda", dtype=torch.float16)
    weight = torch.randn(96, 64, device="cuda", dtype=torch.float16) / 8.0
    bias = torch.randn(96, device="cuda", dtype=torch.float16) / 8.0 if has_bias else None

    actual = qwen_o_proj_triton_forward(x, weight, bias)
    expected = F.linear(x, weight, bias)

    torch.testing.assert_close(actual, expected, rtol=1.0e-2, atol=1.0e-2)


def test_qwen_o_proj_triton_rejects_bad_decode_shape_on_cuda_if_available():
    if not (torch.cuda.is_available() and TRITON_AVAILABLE):
        pytest.skip("CUDA/Triton required")
    x = torch.randn(2, 2, 8, device="cuda", dtype=torch.float16)
    weight = torch.randn(8, 8, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match=r"\[batch, 1, hidden\]"):
        qwen_o_proj_triton_forward(x, weight)
