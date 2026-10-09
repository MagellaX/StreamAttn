import inspect
import types

import pytest
import torch

from benchmarks import profile_adaptive_executor_attribution as profile
from stream_attention.backends.sm90.transposed_gqa_exact_sources import CPP_SOURCE, CUDA_SOURCE


def test_state_geometry_preserves_records_but_coarsens_finalization():
    current = profile.state_geometry([362, 500])
    grouped = profile.state_geometry([362, 500], records_per_cta=4)
    assert current["valid_records"] == grouped["valid_records"] == 862
    assert current["padded_producer_ctas"] == 1000
    assert grouped["valid_producer_tasks"] == 216
    assert grouped["padded_producer_ctas"] == 250
    assert grouped["allocated_partial_bytes"] == 1_032_000


@pytest.mark.parametrize("rows,c", [([], 1), ([0, 1], 1), ([1], 0)])
def test_state_geometry_rejects_empty_or_invalid_work(rows, c):
    with pytest.raises(ValueError):
        profile.state_geometry(rows, records_per_cta=c)


def test_fixed_support_reference_preserves_holes_and_group_sharing():
    torch.manual_seed(42)
    q = torch.randn(1, 1, 16, 4)
    k, v = torch.randn(1, 96, 2, 4), torch.randn(1, 96, 2, 4)
    rows = [[0, 2], [1]]
    result = profile.selected_reference(q, k, v, rows)
    expected = []
    for h in range(16):
        ids = torch.tensor([j for b in rows[h // 8] for j in range(b * 32, (b + 1) * 32)])
        scores = k[0, ids, h // 8].double() @ q[0, 0, h].double() / 2
        expected.append(scores.softmax(0) @ v[0, ids, h // 8].double())
    torch.testing.assert_close(result, torch.stack(expected), rtol=1e-13, atol=1e-13)
    assert profile.support_hash(rows) != profile.support_hash([[0], [1]])


@pytest.mark.parametrize("rows", [[[0, 0], [1]], [[2, 0], [1]], [[], [1]], [[3], [1]]])
def test_support_reference_rejects_multiplicity_or_bounds_errors(rows):
    q = torch.zeros(1, 1, 16, 4)
    kv = torch.zeros(1, 96, 2, 4)
    with pytest.raises(ValueError):
        profile.selected_reference(q, kv, kv, rows)


def test_phase_runner_uses_the_same_prepared_states():
    calls = []
    plan = types.SimpleNamespace(query_group="q", cache=types.SimpleNamespace(key="k", value="v"),
        routes=types.SimpleNamespace(row_ptr="rows", physical_page_ids="pages",
            active_head_masks="heads", token_valid_masks="tokens"),
        workspace={"partial_o": "o", "partial_lse": "lse"}, output_group="out",
        output="result", max_routes_per_row=512)
    run = profile.phase_runner(plan, lambda *args: calls.append(args), 1)
    assert run() == "result"
    assert calls == [("q", "k", "v", "rows", "pages", "heads", "tokens", "o", "lse", "out", 512, 1)]
    with pytest.raises(ValueError):
        profile.phase_runner(plan, None, 3)


def test_attribution_has_no_search_and_no_additive_latency_claim():
    source = inspect.getsource(profile)
    assert "frontier(" not in source and "from_pretrained" not in source
    assert profile.METHODS == ("full", "contribution_triangle")
    assert "component_timings_are_additive=False" in source
    assert source.index("producer()") < source.index("torch.testing.assert_close(merge()")
    assert "selected_nhd_phase_out" in CPP_SOURCE
    assert "if (phase != 2)" in CUDA_SOURCE and "if (phase != 1)" in CUDA_SOURCE
    assert "int64_t phase = 0" in CUDA_SOURCE


def test_runner_reuses_the_hashed_capture_and_saved_schedule(tmp_path, monkeypatch):
    from benchmarks import run_lightning_adaptive_two_gate as runner
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "a1be014\n")
    args = types.SimpleNamespace(experiment="executor_attribution",
        capture_artifacts=[profile.ROOT / "artifacts/gate0/adaptive_qwen32k_feasibility_lightning_h100_20261009.json"],
        teamspace_id="teamspace", cloud_account="cloud", output_json=tmp_path / "result.json")
    command, source = runner.job_command(args)
    assert source["schema"] == profile.SCHEMA
    assert source["capture_inputs"][0]["sha256"] == "304482bd0d1a1ed319dcb6f6e1a312506f9d48ce6b3670756bb06eaeface6e20"
    assert "profile_adaptive_executor_attribution.py" in command
    assert "--source-report" in command and "Capture hash mismatch" in command
    assert "from_pretrained" not in command and "LIGHTNING_API_KEY" not in command


def test_grouped_runner_uses_the_same_capture_and_runs_gpu_equivalence(tmp_path, monkeypatch):
    from benchmarks import run_lightning_adaptive_two_gate as runner
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "55e5741\n")
    args = types.SimpleNamespace(experiment="executor_grouped",
        capture_artifacts=[profile.ROOT / "artifacts/gate0/adaptive_qwen32k_feasibility_lightning_h100_20261009.json"],
        teamspace_id="teamspace", cloud_account="cloud", output_json=tmp_path / "result.json")
    command, source = runner.job_command(args)
    assert source["schema"] == "streamattn.adaptive_executor_grouped.v1"
    assert "--grouped" in command
    assert "pytest -q tests/test_adaptive_executor_grouped_gpu.py" in command
    assert source["capture_inputs"][0]["sha256"] == "304482bd0d1a1ed319dcb6f6e1a312506f9d48ce6b3670756bb06eaeface6e20"


@pytest.mark.parametrize("grouping", [0, 2, True, 4.0])
def test_selected_grouping_rejects_unsupported_values_before_allocation(grouping):
    from stream_attention.paged import PagedSelectedDecodePlan
    with pytest.raises(ValueError, match="records_per_cta"):
        PagedSelectedDecodePlan.build(None, None, None, schedule_epoch=1, records_per_cta=grouping)


def test_grouped_source_changes_record_identity_and_not_online_state_algebra():
    assert "next_record = selected_route + next_tile" in CUDA_SOURCE
    assert "current_record = selected_route + tile" in CUDA_SOURCE
    assert "route_active_head_masks[current_record * 4 + atom]" in CUDA_SOURCE
    assert "route_token_valid_masks[current_record * 4 + atom]" in CUDA_SOURCE
    assert "selected_nhd_grouped_out" in CPP_SOURCE
    assert "fragmented_impl<true, 4>" in CUDA_SOURCE
    assert "selected merge supports at most 512 partitions" in CUDA_SOURCE
