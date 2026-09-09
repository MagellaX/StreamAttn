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
