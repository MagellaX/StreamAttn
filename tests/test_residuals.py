import torch

from stream_attention.residuals import (
    construct_query_exact_residual,
    merge_normalized_attention_states,
    merge_seed_with_residual,
)


def _state(q, k, v):
    scores = torch.einsum("thd,thnd->thn", q, k) / (q.shape[-1] ** 0.5)
    log_z = torch.logsumexp(scores, dim=-1)
    output = torch.einsum("thn,thnd->thd", torch.softmax(scores, dim=-1), v)
    return log_z, output


def test_query_exact_residual_reconstructs_full_attention():
    torch.manual_seed(3)
    q = torch.randn(3, 2, 8)
    seed_k = torch.randn(3, 2, 5, 8)
    seed_v = torch.randn_like(seed_k)
    omitted_k = torch.randn(3, 2, 7, 8)
    omitted_v = torch.randn_like(omitted_k)

    seed_log_z, seed_output = _state(q, seed_k, seed_v)
    omitted_log_z, omitted_output = _state(q, omitted_k, omitted_v)
    residual_k, residual_v = construct_query_exact_residual(q, omitted_log_z, omitted_output)
    reconstructed = merge_seed_with_residual(
        q,
        seed_log_z,
        seed_output,
        residual_k,
        residual_v,
    )

    full_k = torch.cat([seed_k, omitted_k], dim=2)
    full_v = torch.cat([seed_v, omitted_v], dim=2)
    _full_log_z, full_output = _state(q, full_k, full_v)
    torch.testing.assert_close(reconstructed, full_output, rtol=2e-5, atol=2e-5)


def test_shared_true_gqa_residual_bank_expands_by_kv_group():
    q = torch.tensor([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]])
    seed_log_z = torch.zeros(1, 4)
    seed_output = torch.zeros_like(q)
    residual_k = torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]])
    residual_v = torch.tensor([[[3.0, 0.0]], [[0.0, 5.0]]])

    output = merge_seed_with_residual(q, seed_log_z, seed_output, residual_k, residual_v)
    assert output[0, 0, 0] > 2.0
    torch.testing.assert_close(output[0, 0], output[0, 1])
    torch.testing.assert_close(output[0, 2], output[0, 3])
    assert output[0, 2, 1] > 3.0


def test_residual_merge_validates_true_gqa_divisibility():
    q = torch.zeros(1, 3, 4)
    with torch.no_grad():
        try:
            merge_seed_with_residual(
                q,
                torch.zeros(1, 3),
                torch.zeros_like(q),
                torch.zeros(2, 1, 4),
                torch.zeros(2, 1, 4),
            )
        except ValueError as exc:
            assert "divisible" in str(exc)
        else:
            raise AssertionError("expected true-GQA shape validation")


def test_merge_normalized_attention_states_matches_concatenation():
    torch.manual_seed(7)
    q = torch.randn(2, 3, 4)
    left_k = torch.randn(2, 3, 5, 4)
    left_v = torch.randn_like(left_k)
    right_k = torch.randn(2, 3, 6, 4)
    right_v = torch.randn_like(right_k)
    left_log_z, left_output = _state(q, left_k, left_v)
    right_log_z, right_output = _state(q, right_k, right_v)
    merged = merge_normalized_attention_states(
        left_log_z, left_output, right_log_z, right_output
    )
    _full_log_z, full_output = _state(
        q,
        torch.cat([left_k, right_k], dim=2),
        torch.cat([left_v, right_v], dim=2),
    )
    torch.testing.assert_close(merged, full_output, rtol=2e-5, atol=2e-5)
