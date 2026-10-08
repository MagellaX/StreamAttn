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


def test_capture_only_uses_fresh_forwards_and_only_last_query(monkeypatch):
    import sys
    from benchmarks import profile_adaptive_real_activations as capture_module
    from benchmarks import profile_real_llm_gate1_heads as hooks

    forwards = []

    class Tokens(dict):
        def __getattr__(self, name):
            return self[name]

        def to(self, device):
            return self

    class Tokenizer:
        truncation_side = "right"

        def __call__(self, text, **kwargs):
            assert self.truncation_side == "left"
            assert kwargs["max_length"] == 64
            return Tokens(input_ids=torch.ones(1, 64, dtype=torch.int64),
                          attention_mask=torch.ones(1, 64, dtype=torch.int64))

    class Model:
        config = types.SimpleNamespace(max_position_embeddings=64, num_hidden_layers=36,
                                       _commit_hash=REVISION)

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            assert kwargs["use_cache"] is False
            forwards.append(kwargs["input_ids"].shape[1])

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())
    transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *a, **kw: Model())
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setattr(hooks, "_capture_attention_inputs", lambda model, layers: (
        [types.SimpleNamespace(layer_id=layer) for layer in sorted(layers)], []))
    monkeypatch.setattr(hooks, "_shape_qkv", lambda *a, **kw: (
        *(torch.zeros(1, 64, 16, 4, dtype=torch.bfloat16) for _ in range(3)),
        {"rope_applied": True, "q_per_kv": 8}))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "mock")
    monkeypatch.setattr(capture_module, "evaluate", lambda *a, **kw: pytest.fail(
        "capture-only must not run the old multi-query diagnostic"))
    metadata, captures = capture_module.capture_and_evaluate(MODEL, 64, revision=REVISION,
        layers=(0, 16), capture_only=True, last_query_only=True)
    assert forwards == [64, 64]
    assert len(captures) == 4 and metadata["capture_only"] is True
    assert {c["prompt_id"] for c in captures} == set(PROMPTS)
    for c in captures:
        assert c["q"].shape == (1, 1, 16, 4)
        assert c["k"].shape == (1, 64, 2, 4)
        assert c["query_positions"].tolist() == [63]


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


def test_native_conditions_resolve_different_winners_and_normalize_calls(monkeypatch):
    from benchmarks import micro_prefill_baselines as baselines
    from benchmarks import profile_paged_exact_decode as paged_benchmark
    from benchmarks import profile_sm90_micro_prefill as timing
    from benchmarks.profile_adaptive_feasibility import native_headroom
    from stream_attention import paged, selected_routes
    from stream_attention import planning

    original_to, original_arange, original_tensor = torch.Tensor.to, torch.arange, torch.tensor

    def cpu_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    def cpu_factory(factory):
        def invoke(*args, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            return factory(*args, **kwargs)
        return invoke

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch, "arange", cpu_factory(original_arange))
    monkeypatch.setattr(torch, "tensor", cpu_factory(original_tensor))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    trace, prepared = [], []

    def run_factory(name, q):
        def run():
            trace.append((name, q.data_ptr()))
            return torch.zeros_like(q)
        return run

    def prepare(q, k, v):
        prepared.append(q.data_ptr())
        return {name: run_factory(name, q) for name in ("flashinfer_a", "flashinfer_b")}, {}

    monkeypatch.setattr(baselines, "prepare_baselines", prepare)
    monkeypatch.setattr(baselines, "baseline_versions", lambda: {"test": "mock"})
    monkeypatch.setattr(paged_benchmark, "_flashinfer_runner", lambda *a, **kw: (_ for _ in ()).throw(ValueError("mock")))
    monkeypatch.setattr(paged, "PagedKVCache", lambda *args: types.SimpleNamespace())

    class Exact:
        @staticmethod
        def build(q, cache):
            return types.SimpleNamespace(run=run_factory("native", q), backend="test")

    class Selected:
        @staticmethod
        def build(q, cache, routes, **kwargs):
            return types.SimpleNamespace(run=run_factory("selected", q), producer_ctas=2,
                max_routes_per_row=1, workspace_bytes=256)

    monkeypatch.setattr(paged, "PagedExactDecodePlan", Exact)
    monkeypatch.setattr(paged, "PagedSelectedDecodePlan", Selected)
    monkeypatch.setattr(planning.AttentionProblem, "from_paged", lambda *a, **kw: None)
    monkeypatch.setattr(planning.AttentionTilePlan, "selected", lambda *a, **kw: None)
    monkeypatch.setattr(selected_routes, "prepare_paged_routes64", lambda *a: types.SimpleNamespace(
        route_count=2, metadata_bytes=64, group_route_efficiency=1.0))
    monkeypatch.setattr(timing, "_capture", lambda run, **kw: run)

    def elapsed(graph, **kwargs):
        trace.clear()
        graph()
        rotating = len({ptr for _, ptr in trace}) > 1
        name = trace[0][0]
        per_call = {"flashinfer_a": 2 if rotating else 1,
                    "flashinfer_b": 1 if rotating else 2,
                    "native": 4, "selected": 3}[name]
        return len(trace) * per_call

    monkeypatch.setattr(timing, "_elapsed_graph_ms", elapsed)
    q = torch.zeros(1, 1, 8, 64, dtype=torch.bfloat16)
    k = torch.zeros(1, 32, 1, 64, dtype=torch.bfloat16)
    diagnostic, full, refs = frontier(q, k, k, torch.tensor([31]))
    result = native_headroom(q, k, k, diagnostic, full, refs, buffer_copies=8, trials=3)
    assert len(set(prepared)) == 8
    assert result["conditions"]["warm_fixed_buffer"]["fastest_tested_correct_exact"] == "flashinfer_a"
    assert result["conditions"]["rotating_working_set"]["fastest_tested_correct_exact"] == "flashinfer_b"
    for condition in result["conditions"].values():
        assert condition["calls_per_graph"] == 8
        for schedule in condition["selected"]:
            assert schedule["correctness"]["replicas_checked"] == 8
            assert schedule["headroom_ms"] == -2
            assert schedule["paired"][0]["exact"] == 1
