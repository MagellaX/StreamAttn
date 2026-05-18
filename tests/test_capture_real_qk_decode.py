import torch

from benchmarks.capture_real_qk_decode import _crop_decode_tensors, _safe_name


def test_crop_decode_tensors_keeps_last_query_and_kv_window():
    q = torch.arange(1 * 5 * 2 * 3).reshape(1, 5, 2, 3)
    k = torch.arange(1 * 10 * 2 * 3).reshape(1, 10, 2, 3)
    v = k + 1000

    q_out, k_out, v_out = _crop_decode_tensors(q, k, v, query_len=1, kv_len=4)

    assert torch.equal(q_out, q[:, -1:, :, :])
    assert torch.equal(k_out, k[:, -4:, :, :])
    assert torch.equal(v_out, v[:, -4:, :, :])


def test_safe_name_strips_repo_prefix_and_symbols():
    assert _safe_name("HuggingFaceTB/SmolLM2-135M") == "SmolLM2-135M"
    assert _safe_name("owner/model with spaces") == "model_with_spaces"
