import pytest
import torch

from benchmarks.profile_adaptive_feasibility import frontier


def sample(values, *, heads=8):
    values = torch.tensor(values, dtype=torch.float64).view(1, -1, 1, 1)
    q = torch.zeros(1, 1, heads, 1, dtype=torch.float64)
    k = torch.zeros_like(values)
    return q, k, values, torch.tensor([values.shape[1] - 1])


def row(result, method):
    return next(r for r in result["schedules"] if r["method"] == method)


def test_value_contribution_exposes_slack_hidden_by_mass_radius():
    args = sample([1, -1] * 8)
    result, _, _ = frontier(*args, budget=1e-3, block_size=2)
    assert row(result, "exact_mass_radius")["omitted_token_fraction"] == 0
    assert row(result, "contribution_triangle")["omitted_token_fraction"] == 7 / 8
    assert row(result, "contribution_triangle")["max_output_l2_error"] == 0


def test_global_budget_is_not_reset_per_block():
    result, full, references = frontier(*sample([0, 0, 1, 1, 2, 2, 4, 4]),
                                        budget=0.15, block_size=2)
    for r in result["schedules"]:
        assert r["max_output_l2_error"] <= 0.15 + 1e-12
        assert (references[r["method"]] - full).norm(dim=-1).max() <= 0.15 + 1e-12
        assert r["globally_optimal"] is False


def test_shared_group_obeys_sensitive_head():
    q, k, v, positions = sample([0, 0, 1, 1, 2, 2, 3, 3])
    q[0, 0, -1] = 10
    k.copy_(v)
    result, full, references = frontier(q, k, v, positions, budget=0.01, block_size=2)
    for r in result["schedules"]:
        assert len(r["kept_blocks"]) == 1
        assert (references[r["method"]] - full).norm(dim=-1).max() <= 0.01 + 1e-12


def test_zero_budget_is_no_omission_even_with_identical_values():
    result, _, _ = frontier(*sample([2] * 8), budget=0, block_size=2)
    assert all(r["kept_blocks"] == [[0, 1, 2, 3]] for r in result["schedules"])


def test_tail_is_counted_and_schedule_is_nonempty():
    result, _, _ = frontier(*sample([2] * 7), budget=1e-3, block_size=2)
    for r in result["schedules"]:
        assert r["kept_blocks"][0]
        assert 0 <= r["omitted_token_fraction"] < 1
    assert row(result, "full")["retained_tokens_per_kv_head"] == [7]


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_invalid_budget(budget):
    with pytest.raises(ValueError):
        frontier(*sample([1, 2]), budget=budget)


def test_partial_visibility_is_rejected_not_silently_timed_as_decode():
    q, k, v, positions = sample([1, 2, 3, 4])
    with pytest.raises(ValueError, match="all-visible"):
        frontier(q, k, v, positions - 1)


def test_nonfinite_activations_rejected():
    q, k, v, positions = sample([1, 2])
    v[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite activations"):
        frontier(q, k, v, positions)
