import pytest
import torch

from benchmarks.profile_adaptive_real_activations import evaluate, physical_vote


def fixture():
    q = torch.zeros(1, 3, 4, 4)
    q[..., 0] = 4
    k = torch.zeros(1, 8, 2, 4)
    k[:, :4, :, 0] = 4
    k[:, 4:, :, 0] = -4
    v = torch.arange(64).reshape(1, 8, 2, 4).float() / 16
    return q, k, v, torch.arange(5, 8)


def test_centered_radius_translation_invariance_and_cumulative_budget():
    q, k, v, positions = fixture()
    original = evaluate(q, k, v, positions, block_size=4)
    shifted = evaluate(q, k, v + 64, positions, block_size=4)
    assert shifted["radius_origin"][0] > original["radius_origin"][0]
    assert shifted["radius_mean_centered"] == original["radius_mean_centered"]
    for a, b in zip(original["results"], shifted["results"]):
        assert a["max_error"] <= a["max_bound"] + 1e-9
        assert a["max_bound"] <= original["budget"] + 1e-12
        if a["radius"] == "mean_centered":
            assert a["pre_rows"] == b["pre_rows"] and a["post_rows"] == b["post_rows"]
            assert a["max_error"] == pytest.approx(b["max_error"], abs=1e-12)


def test_constant_values_have_zero_centered_omission_error():
    q, k, v, positions = fixture()
    result = evaluate(q, k, v.fill_(7), positions, block_size=4)
    assert result["radius_mean_centered"] == [0, 0]
    centered = [r for r in result["results"] if r["radius"] == "mean_centered"]
    assert any(r["post_rows"] > 0 for r in centered)
    assert all(r["max_error"] < 1e-12 and r["max_bound"] == 0 for r in centered)


def test_votes_respect_head_groups_and_inactive_rows():
    valid = torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    proposed = valid.clone()
    proposed[1, 0] = False
    expected = valid.clone()
    expected[:2] = False
    assert torch.equal(physical_vote(proposed, valid, 2, 4), expected)


def test_append_visibility_must_be_explicit_and_in_range():
    q, k, v, positions = fixture()
    with pytest.raises(ValueError, match="query positions"):
        evaluate(q, k, v, positions + 8, block_size=4)


def test_zero_budget_grouping_charges_duplicate_kv_reads():
    q, k, v, positions = fixture()
    result = evaluate(q, k, v, positions, budget=0, block_size=4,
                      head_groups=[1, 2], query_tiles=[1, 2, 4])
    for row in result["results"]:
        assert row["pre_regions"] == row["post_regions"] == 0
        assert row["kv_read_ratio_vs_full_group"] == 2 / row["head_group_size"]
        assert row["requested_k_bytes"] == row["requested_v_bytes"]
        expected = 2 * k.shape[2] * k.shape[1] * k.shape[-1] * k.element_size()
        expected *= (q.shape[1] + row["query_tile_size"] - 1) // row["query_tile_size"]
        assert row["exact_full_group_kv_bytes"] == expected


def test_work_accounting_distinguishes_qk_and_pv_savings():
    q, k, v, positions = fixture()
    result = evaluate(q, k, v, positions, block_size=4, head_groups=[1, 2])
    for row in result["results"]:
        assert row["requested_v_bytes"] <= row["requested_k_bytes"]
        if row["decision"] == "exact_mass_oracle":
            assert row["pre_regions"] == 0
            assert row["requested_k_bytes"] * 2 == row["exact_full_group_kv_bytes"] * 2 / row["head_group_size"]
        assert row["max_bound"] <= result["budget"] + 1e-12


@pytest.mark.parametrize("groups,tiles", [([3], [1]), ([0], [1]), ([1], [0]), ([], [1]), ([1], [])])
def test_invalid_physical_scope_is_rejected(groups, tiles):
    with pytest.raises(ValueError, match="head groups"):
        evaluate(*fixture(), head_groups=groups, query_tiles=tiles)


def test_repeated_scopes_do_not_duplicate_evidence():
    result = evaluate(*fixture(), block_size=4, head_groups=[1, 1, 2], query_tiles=[1, 1])
    assert len(result["results"]) == 12


def test_requested_traffic_counts_partial_tail_and_invisible_blocks():
    q, k, v, _ = fixture()
    k, v = torch.cat((k, k[:, :1]), 1), torch.cat((v, v[:, :1]), 1)
    result = evaluate(q, k, v, torch.tensor([0, 5, 8]), budget=0,
                      block_size=4, head_groups=[1, 2], query_tiles=[2])
    # First query tile touches eight keys, second touches nine, per KV head.
    expected = (8 + 9) * 2 * 4 * 4 * 2
    for row in result["results"]:
        assert row["exact_full_group_kv_bytes"] == expected
        assert row["requested_k_bytes"] + row["requested_v_bytes"] == expected * 2 / row["head_group_size"]


def test_post_only_subgroups_cannot_beat_full_group_requested_bytes():
    q, k, v, positions = fixture()
    result = evaluate(q, k, torch.zeros_like(v), positions, block_size=4,
                      head_groups=[1, 2], query_tiles=[1, 4])
    for row in result["results"]:
        if row["decision"] == "exact_mass_oracle" and row["head_group_size"] == 1:
            assert row["post_regions"] > 0
            assert row["kv_read_ratio_vs_full_group"] >= 1
