"""Keep failed cases and physical-work limitations visible in adaptive reports."""

import pytest

from benchmarks.summarize_adaptive_two_gate import summarize


def artifact():
    return dict(
        schema="streamattn.adaptive_two_gate.v1", provider="local", device="test",
        complete=False, failures=[dict(kind="causal", error="original limit failed")],
        cases=[dict(
            kind="mixed_rows", dtype="torch.float16",
            correctness=dict(adaptive=dict(counters=[100, 0, 100, 20, 20, 20],
                                           max_row_l2=0.001, max_omission_bound=0.0001)),
            graph_ms=dict(adaptive=[2.0, 4.0], mask_only_control=[6.0, 2.0]),
            median_graph_ms=dict(adaptive=3.0, mask_only_control=4.0),
            baseline_error="baseline unavailable",
        )],
    )


def test_summary_preserves_failures_and_distinguishes_logical_from_physical_skips():
    source = artifact()
    result = summarize(source)
    assert not result["input_complete"]
    assert not result["performance_promotion"] and not result["model_validation"]
    assert result["failures"] == source["failures"]
    row = result["rows"][0]
    assert row["pre_row_blocks"] == 100
    assert row["executed_qk_tiles"] == row["executed_pv_tiles"] == row["valid_cta_tiles"]
    assert row["baseline_error"] == "baseline unavailable"
    ratio = row["ratios"]["mask_only_control_over_adaptive"]
    assert ratio == dict(median=1.75, minimum=0.5, wins=1, trials=2)
    assert "torch_flash_sdpa_over_adaptive" not in row["ratios"]


def test_summary_rejects_unpaired_timings():
    source = artifact()
    source["cases"][0]["graph_ms"]["mask_only_control"].pop()
    with pytest.raises(ValueError, match="paired timing lengths differ"):
        summarize(source)


@pytest.mark.parametrize("samples", [[], [0.0], [-1.0], [float("nan")], [float("inf")]])
def test_summary_rejects_invalid_timings(samples):
    source = artifact()
    source["cases"][0]["graph_ms"]["adaptive"] = samples
    with pytest.raises(ValueError, match="timings must be"):
        summarize(source)


def test_summary_rejects_other_schemas():
    source = artifact()
    source["schema"] = "different.v1"
    with pytest.raises(ValueError, match="expected adaptive"):
        summarize(source)
