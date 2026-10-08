import types

import pytest
import torch

from benchmarks.profile_adaptive_feasibility import frontier, graph_buffer_indices
from benchmarks.profile_adaptive_long_context import (
    LAYERS, MODEL, PROMPTS, REVISION, append_current_certificate, validate_capture_contract,
)
from benchmarks.profile_adaptive_real_activations import evaluate
from benchmarks import run_lightning_adaptive_two_gate as runner


def test_matching_graph_call_counts_and_distinct_rotation_indices():
    assert graph_buffer_indices(8, "warm_fixed_buffer") == [0] * 8
    assert graph_buffer_indices(8, "rotating_working_set") == list(range(8))


@pytest.mark.parametrize("copies,condition", [(0, "warm_fixed_buffer"),
    (1, "rotating_working_set"), (8, "unknown")])
def test_invalid_graph_construction_rejected(copies, condition):
    with pytest.raises(ValueError):
        graph_buffer_indices(copies, condition)


def contract_fixture():
    # Expanded views validate the archive contract without allocating ten 32K caches.
    q = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16).expand(1, 1, 16, 128)
    k = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16).expand(1, 32768, 2, 128)
    captures = [dict(prompt_id=p, layer=layer, q=q, k=k, v=k,
        query_positions=torch.tensor([32767]), meta={"rope_applied": True})
        for p in PROMPTS for layer in LAYERS]
    return dict(model=MODEL, model_revision=REVISION, max_seq=32768), captures


def test_pinned_capture_contract():
    validate_capture_contract(*contract_fixture())


@pytest.mark.parametrize("change", ["revision", "duplicate", "position", "rope", "dtype"])
def test_capture_contract_rejects_silent_semantic_changes(change):
    metadata, captures = contract_fixture()
    if change == "revision":
        metadata["model_revision"] = "main"
    elif change == "duplicate":
        captures[-1] = captures[0]
    elif change == "position":
        captures[0]["query_positions"] = torch.tensor([8191])
    elif change == "rope":
        captures[0]["meta"]["rope_applied"] = False
    else:
        captures[0]["q"] = captures[0]["q"].float()
    with pytest.raises(ValueError):
        validate_capture_contract(metadata, captures)


def test_current_certificate_exports_shared_support_and_matches_fp64():
    q = torch.zeros(1, 1, 16, 4, dtype=torch.float64)
    q[..., 0] = 8
    k = torch.zeros(1, 96, 2, 4, dtype=torch.float64)
    k[:, :32, :, 0] = 4
    k[:, 32:, :, 0] = -4
    v = torch.randn(k.shape, generator=torch.Generator().manual_seed(47), dtype=torch.float64)
    positions = torch.tensor([95])
    diagnostic, full, refs = frontier(q, k, v, positions, block_size=32)
    current = evaluate(q, k, v, positions, block_size=32,
                       head_groups=(8,), query_tiles=(1,), include_schedules=True)
    append_current_certificate(q, k, v, diagnostic, full, refs, current)
    schedule = diagnostic["schedules"][-1]
    assert schedule["method"] == "current_two_gate"
    assert len(schedule["kept_blocks"]) == 2
    assert schedule["omitted_token_fraction"] > 0
    assert float((refs["current_two_gate"] - full).norm(dim=-1).max()) <= 1e-3
    assert schedule["deployable_selector"] is False


def test_schedule_export_rejects_nonshared_scope():
    with pytest.raises(ValueError, match="full GQA"):
        evaluate(torch.zeros(1, 1, 8, 4), torch.zeros(1, 32, 1, 4),
                 torch.zeros(1, 32, 1, 4), torch.tensor([31]),
                 head_groups=(1,), query_tiles=(1,), include_schedules=True)


def test_combined_runner_freezes_fresh_protocol_and_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "600eda1\n")
    args = types.SimpleNamespace(experiment="long_context_feasibility",
        output_json=tmp_path / "result.json", teamspace_id="test-teamspace", cloud_account="test-cloud")
    command, source = runner.job_command(args)
    assert source["schema"] == "streamattn.adaptive_long_context_feasibility.v1"
    assert source["capture_inputs"] == []
    assert source["capture_remote_path"] == "uploads/streamattn/result.captures.pt"
    assert "profile_adaptive_long_context.py --output-json" in command
    assert "transformers==4.51.3" in command and "flashinfer-python==0.6.13" in command
    assert "STREAMATTN_CUTLASS_ROOT" in command
    assert "test_adaptive_long_context.py" in command
    assert "LIGHTNING_API_KEY" not in command
