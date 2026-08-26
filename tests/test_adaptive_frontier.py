import torch

from stream_attention.adaptive_frontier import (
    BlockAttentionStates,
    control_variate_attention,
    diagonal_gaussian_block_states,
    exact_block_attention_states,
    gqa_topk_mask,
    merge_block_attention_states,
    poisson_tail_sample,
    post_wo_gqa_greedy_mask,
)


def _dense_attention(q, k, v, lengths):
    group_size = q.shape[1] // k.shape[0]
    kq = k.repeat_interleave(group_size, dim=0)
    vq = v.repeat_interleave(group_size, dim=0)
    scores = torch.einsum("rhd,hnd->rhn", q, kq) / (q.shape[-1] ** 0.5)
    positions = torch.arange(k.shape[1])[None, None, :]
    scores = scores.masked_fill(positions >= lengths[:, None, None], -torch.inf)
    return torch.einsum("rhn,hnd->rhd", torch.softmax(scores, dim=-1), vq)


def test_exact_block_states_merge_to_dense_causal_attention():
    torch.manual_seed(1)
    q = torch.randn(3, 4, 8)
    k = torch.randn(2, 11, 8)
    v = torch.randn_like(k)
    lengths = torch.tensor([7, 10, 11])
    states = exact_block_attention_states(
        q, k, v, block_size=4, valid_lengths=lengths
    )
    merged = merge_block_attention_states(states)
    expected = _dense_attention(q, k, v, lengths)
    assert bool(merged.valid.all())
    torch.testing.assert_close(merged.output, expected, rtol=2e-5, atol=2e-5)


def test_diagonal_moment_state_is_exact_for_constant_blocks():
    q = torch.tensor([[[1.0, -0.5], [0.25, 0.75]]])
    k = torch.tensor([[[2.0, 1.0]] * 4, [[-1.0, 3.0]] * 4])
    v = torch.tensor([[[4.0, -2.0]] * 4, [[1.5, 5.0]] * 4])
    exact = exact_block_attention_states(q, k, v, block_size=4)
    moment = diagonal_gaussian_block_states(q, k, v, block_size=4)
    torch.testing.assert_close(moment.log_partition, exact.log_partition)
    torch.testing.assert_close(moment.output, exact.output)


def test_control_variate_limits_recover_approximate_and_exact_states():
    torch.manual_seed(2)
    q = torch.randn(2, 2, 4)
    k = torch.randn(1, 8, 4)
    v = torch.randn_like(k)
    exact = exact_block_attention_states(q, k, v, block_size=2)
    approximate = BlockAttentionStates(
        log_partition=exact.log_partition + 0.3,
        output=exact.output * 0.5,
    )
    none = torch.zeros_like(exact.log_partition, dtype=torch.bool)
    all_blocks = torch.ones_like(none)
    pure_approximate = control_variate_attention(
        exact, approximate, selected=none
    )
    expected_approximate = merge_block_attention_states(approximate)
    torch.testing.assert_close(
        pure_approximate.output, expected_approximate.output, rtol=2e-5, atol=2e-5
    )
    corrected = control_variate_attention(
        exact, approximate, selected=all_blocks
    )
    expected_exact = merge_block_attention_states(exact)
    torch.testing.assert_close(corrected.output, expected_exact.output, rtol=2e-5, atol=2e-5)


def test_unit_probability_sample_correction_recovers_exact_tail():
    torch.manual_seed(3)
    q = torch.randn(1, 2, 4)
    k = torch.randn(1, 6, 4)
    v = torch.randn_like(k)
    exact = exact_block_attention_states(q, k, v, block_size=2)
    approximate = BlockAttentionStates(exact.log_partition - 0.4, exact.output + 0.2)
    selected = torch.zeros_like(exact.log_partition, dtype=torch.bool)
    sampled = torch.ones_like(selected)
    corrected = control_variate_attention(
        exact,
        approximate,
        selected=selected,
        sampled=sampled,
        inclusion_probability=torch.ones_like(exact.log_partition),
    )
    expected = merge_block_attention_states(exact)
    torch.testing.assert_close(corrected.output, expected.output, rtol=2e-5, atol=2e-5)


def test_gqa_topk_uses_one_shared_route_per_kv_group():
    scores = torch.tensor(
        [[[1.0, 5.0, 0.0], [4.0, 0.0, 2.0], [0.0, 1.0, 7.0], [3.0, 2.0, 0.0]]]
    )
    mask = gqa_topk_mask(scores, kv_heads=2, blocks_per_group=1)
    assert mask[0, 0].tolist() == [False, True, False]
    assert torch.equal(mask[0, 0], mask[0, 1])
    assert mask[0, 2].tolist() == [False, False, True]
    assert torch.equal(mask[0, 2], mask[0, 3])


def test_post_wo_oracle_chooses_the_only_useful_shared_block():
    log_z = torch.tensor([[[-2.0, 0.0], [-2.0, 0.0]]])
    outputs = torch.tensor([[[[0.0], [2.0]], [[0.0], [3.0]]]])
    states = BlockAttentionStates(log_z, outputs)
    full = merge_block_attention_states(states).output
    weight = torch.eye(2)
    mask = post_wo_gqa_greedy_mask(
        states,
        full_output=full,
        o_proj_weight=weight,
        kv_heads=1,
        blocks_per_group=1,
    )
    assert mask[0, 0].tolist() == [False, True]
    assert torch.equal(mask[0, 0], mask[0, 1])


def test_poisson_tail_sampling_respects_selection_and_expected_budget():
    priority = torch.ones(2, 2, 8)
    selected = torch.zeros_like(priority, dtype=torch.bool)
    selected[..., 0] = True
    generator = torch.Generator().manual_seed(4)
    sampled, probability = poisson_tail_sample(
        priority,
        selected=selected,
        expected_samples=3,
        generator=generator,
    )
    assert not bool((sampled & selected).any())
    assert torch.all(probability[selected] == 0)
    torch.testing.assert_close(
        probability.sum(dim=-1), torch.full((2, 2), 3.0), rtol=1e-5, atol=1e-5
    )
