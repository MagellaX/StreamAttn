import math
from types import SimpleNamespace

import pytest
import torch

import benchmarks.profile_sm90_micro_prefill_temporal as profiler
from stream_attention.backends.sm90 import micro_prefill_temporal as backend
from benchmarks.profile_sm90_micro_prefill_temporal import (
    experiment_cases,
    parse_args,
    schedule_geometry,
)


def test_anchors_isolate_pipeline_depth_and_parallel_work():
    cases = experiment_cases("anchors")
    a, b, c = [schedule_geometry(row) for row in cases]
    assert a["producer_ctas"] == b["producer_ctas"] == 256
    assert a["tiles_per_cta_mean"] == c["tiles_per_cta_mean"] == 4
    assert b["tiles_per_cta_mean"] == 16
    assert c["producer_ctas"] == 512
    assert a["partial_bytes"] == 8.0625 * 2**20
    assert a["unique_kv_bytes"] == 4 * 2**20
    assert a["logical_kv_bytes"] == 32 * 2**20
    assert a["qk_pv_flops"] == 2147483648


@pytest.mark.parametrize("suite", ["smoke", "anchors", "boundary", "split_diagnostic"])
def test_split_work_product_and_balanced_nonempty_intervals(suite):
    for row in experiment_cases(suite):
        geometry = schedule_geometry(row)
        assert geometry["tiles_per_cta_min"] >= 1
        assert geometry["tiles_per_cta_max"] - geometry["tiles_per_cta_min"] <= 1
        assert math.isclose(
            geometry["producer_ctas"] * geometry["tiles_per_cta_mean"],
            geometry["cta_tile_iterations"],
        )


def test_optional_split_probe_changes_both_variables():
    row = schedule_geometry(experiment_cases("split_diagnostic")[0])
    assert row["producer_ctas"] == 128
    assert row["tiles_per_cta_mean"] == 8


def test_cli_requires_positive_timing_counts(tmp_path):
    args = [
        "--cutlass-root",
        str(tmp_path),
        "--build-dir",
        str(tmp_path),
        "--output",
        str(tmp_path / "result.json"),
    ]
    assert parse_args(args).suite == "smoke"
    with pytest.raises(SystemExit):
        parse_args(args + ["--repeats", "0"])
    with pytest.raises(ValueError):
        experiment_cases("broad_sweep")


@pytest.fixture
def cpu_profiler(monkeypatch, tmp_path):
    """Exercise real profiler control flow; CUDA plans/graphs run as CPU doubles."""
    import benchmarks.sm90_binary_diagnostics as diagnostics

    case = dict(batch=1, m=2, n=64, hq=4, g=4, d=64, splits=1)
    args = parse_args(
        [
            "--cutlass-root",
            str(tmp_path),
            "--build-dir",
            str(tmp_path / "build"),
            "--output",
            str(tmp_path / "result.json"),
            "--warmup",
            "1",
            "--iterations",
            "2",
            "--repeats",
            "3",
        ]
    )
    state = SimpleNamespace(
        case=case,
        args=args,
        events=[],
        builds={},
        plans={},
        graphs={},
        diagnostics={},
        failing_protocol=None,
        resource_failure="spill",
        stale_graph=None,
        baseline_bad=False,
        timing_order=[],
    )
    original_empty, original_generator = torch.empty, torch.Generator
    reference = profiler.fp32_reference

    def cpu_empty(*shape, **kwargs):
        kwargs["device"] = "cpu"
        return original_empty(*shape, **kwargs)

    def cpu_generator(**kwargs):
        state.events.append("initialize")
        return original_generator(device="cpu")

    def checked_reference(q, k, v):
        assert not q.is_cuda and not k.is_cuda and not v.is_cuda
        state.events.append("reference")
        return reference(q, k, v)

    class Plan:
        def __init__(self, name, q, k, v, kwargs):
            self.name = name
            self.query, self.key_cache, self.value_cache = q, k, v
            self.output = torch.empty_like(q)
            self.query_tiles = 1
            self.num_splits = kwargs["num_splits"]
            self.partial_output = original_empty(1, 1, 64, 64)
            self.partial_lse = original_empty(1, 1, 64)
            self.workspace_bytes = 64 * 65 * 4
            self.extension = SimpleNamespace(
                __file__=str(kwargs["build_dir"] / "test_extension.so"),
                natural_micro_prefill_components_out=lambda *a: self.run_component(
                    "producer" if a[-1] == 1 else "merge"
                ),
            )
            if name != "control64":
                self.resources = backend.decode_resource_info(
                    torch.tensor(
                        [
                            106,
                            0,
                            32768,
                            0,
                            4,
                            128,
                            32,
                            2056,
                            0,
                            0,
                            16,
                            128,
                        ]
                    )
                )
                if name == state.failing_protocol:
                    if state.resource_failure == "spill":
                        self.resources["producer"]["local_bytes_per_thread"] = 8
                    else:
                        self.resources["producer"]["blocks_per_sm"] = 0

        @property
        def resource_pass(self):
            assert self.name != "control64", (
                "original control has no runtime resource API"
            )
            state.events.append("gate:" + self.name)
            return backend.resource_gate(self.resources)["passed"]

        def run_component(self, which):
            state.events.append(self.name + ":" + which)
            rows = self.query.shape[1] * self.query.shape[2]
            if which == "producer":
                output, lse = reference(self.query, self.key_cache, self.value_cache)
                self.partial_output.zero_()
                self.partial_lse.zero_()
                self.partial_output[0, 0, :rows].copy_(output.reshape(rows, 64))
                self.partial_lse[0, 0, :rows].copy_(lse.reshape(rows) / math.log(2))
            else:
                assert which == "merge"
                self.output.copy_(
                    self.partial_output[0, 0, :rows].reshape(self.query.shape)
                )
            return self.output

        def run(self):
            self.run_component("producer")
            return self.run_component("merge")

    def build(q, k, v, **kwargs):
        name = kwargs.get("protocol", "control64")
        state.events.append("build:" + name)
        state.builds[name] = kwargs
        plan = Plan(name, q, k, v, kwargs)
        state.plans[name] = plan
        return plan

    def inspect_extension(extension, output_dir, **kwargs):
        name = next(
            name for name, plan in state.plans.items() if plan.extension is extension
        )
        state.events.append("diagnostics:" + name)
        state.diagnostics[name] = kwargs
        return dict(selection="exact_kernel_names", kernel_names=kwargs["kernel_names"])

    class Graph:
        def __init__(self, name, run):
            self.name, self.run = name, run

        def replay(self):
            state.events.append("replay:" + self.name)
            if self.name != state.stale_graph:
                self.run()

    def capture(run, **kwargs):
        state.events.append("capture")
        run()
        owner = getattr(run, "__self__", None)
        if isinstance(owner, Plan):
            name = owner.name
        elif run.__name__ == "baseline":
            name = "torch_flash"
        else:
            name = state.events[-1]
        graph = Graph(name, run)
        state.graphs[name] = graph
        return graph

    def elapsed(graph, **kwargs):
        state.timing_order.append(graph.name)
        graph.replay()
        return dict(control64=0.03, drained=0.02, temporal=0.015, torch_flash=0.01).get(
            graph.name, 0.003
        )

    def flash(q, k, v):
        state.events.append("flash")
        output = reference(q, k, v)[0].to(q.dtype)
        return output + 20 if state.baseline_bad else output

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch, "Generator", cpu_generator)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(profiler.NaturalMicroPrefillPlan, "build", build)
    monkeypatch.setattr(backend.TemporalMicroPrefillPlan, "build", build)
    monkeypatch.setattr(diagnostics, "inspect_extension", inspect_extension)
    monkeypatch.setattr(profiler, "fp32_reference", checked_reference)
    monkeypatch.setattr(profiler, "_flash_sdpa", flash)
    monkeypatch.setattr(profiler, "_capture", capture)
    monkeypatch.setattr(profiler, "_elapsed_graph_ms", elapsed)
    return state


@pytest.mark.parametrize("mode", ["resources", "correctness", "benchmark"])
@pytest.mark.parametrize("protocol", ["drained", "temporal"])
@pytest.mark.parametrize("failure", ["spill", "zero_residency"])
def test_bad_resources_reject_before_initialization_or_execution(
    cpu_profiler, mode, protocol, failure
):
    state = cpu_profiler
    state.args.mode = mode
    state.failing_protocol = protocol
    state.resource_failure = failure
    row = profiler.profile_case(state.case, state.args)
    assert row["status"] == "rejected_resources" and row["resource_pass"] is False
    assert row["resources"]["control64"] is None
    assert not backend.resource_gate(row["resources"][protocol])["passed"]
    assert list(state.plans) == ["control64", "drained", "temporal"]
    assert list(state.diagnostics) == ["control64", "drained", "temporal"]
    assert "initialize" not in state.events and "reference" not in state.events
    assert "flash" not in state.events and "capture" not in state.events
    assert not any(event.endswith((":producer", ":merge")) for event in state.events)
    assert not state.graphs and not state.timing_order
    assert "accuracy" not in row and "median_ms" not in row


def test_resource_only_collects_exact_symbols_without_initialization(cpu_profiler):
    state = cpu_profiler
    state.args.mode = "resources"
    row = profiler.profile_case(state.case, state.args)
    assert row["status"] == "resources_only" and row["resource_pass"] is True
    expected = {
        "control64": "streamattn_natural_wgmma_micro_prefill_partial_kernel",
        "drained": "streamattn_temporal_micro_prefill_partial_kernel<true>",
        "temporal": "streamattn_temporal_micro_prefill_partial_kernel<false>",
    }
    for name, producer in expected.items():
        diagnostic = state.diagnostics[name]
        assert diagnostic["kernel_names"] == dict(
            producer=producer,
            merge="streamattn_natural_wgmma_micro_prefill_merge_kernel",
        )
        assert diagnostic["include_archive"] is False
        assert diagnostic["runtime_resources"] == row["resources"][name]
        assert diagnostic["build_metadata"]["keep_intermediates"] is (
            name != "control64"
        )
    assert state.events.index("diagnostics:temporal") < state.events.index(
        "gate:drained"
    )
    assert state.builds["drained"]["build_dir"] == state.builds["temporal"]["build_dir"]
    assert state.builds["drained"]["diagnostic_build"] is True
    assert "initialize" not in state.events and not state.graphs


def test_resource_gate_also_applies_without_binary_diagnostics(cpu_profiler):
    state = cpu_profiler
    state.args.binary_diagnostics = False
    state.failing_protocol = "temporal"
    row = profiler.profile_case(state.case, state.args)
    assert row["status"] == "rejected_resources"
    assert row["binary_diagnostics"] == {} and not state.diagnostics
    assert state.builds["temporal"]["diagnostic_build"] is False
    assert "initialize" not in state.events


@pytest.mark.parametrize("stale", [False, True])
def test_flash_replay_eligibility_controls_only_flash_ratios(cpu_profiler, stale):
    state = cpu_profiler
    if stale:
        state.stale_graph = "torch_flash"
    row = profiler.profile_case(state.case, state.args)
    assert row["status"] == "passed_canary" and row["exact_pass"]
    assert row["baseline_accuracy"]["passed"]
    assert row["baseline_mutation"]["passed"] is not stale
    assert row["paired_speedup_vs_control"]["temporal"] == pytest.approx([2.0] * 3)
    assert row["paired_speedup_vs_control"]["drained"] == pytest.approx([1.5] * 3)
    if stale:
        assert row["paired_speedup_vs_flash"] == {}
    else:
        assert set(row["paired_speedup_vs_flash"]) == {
            "control64",
            "drained",
            "temporal",
        }
        assert row["paired_speedup_vs_flash"]["control64"] == pytest.approx([1 / 3] * 3)
    assert all(row["composition"].values())
    assert all(
        check["allocated_bytes_delta"] == 0 for check in row["mutation"].values()
    )
    assert state.events.index("gate:temporal") < state.events.index("initialize")
    assert len(row["trials_ms"]) == state.args.repeats
    # Preserve component timing and input-mutation replay instead of a seed-only check.
    assert all(
        f"{name}:{which}" in row["median_ms"]
        for name in state.plans
        for which in ("producer", "merge")
    )


def test_failed_eager_flash_is_not_captured_or_compared(cpu_profiler):
    state = cpu_profiler
    state.baseline_bad = True
    row = profiler.profile_case(state.case, state.args)
    assert row["exact_pass"] and not row["baseline_accuracy"]["passed"]
    assert "torch_flash" not in state.graphs and "torch_flash" not in row["median_ms"]
    assert row["paired_speedup_vs_flash"] == {} and "baseline_mutation" not in row


def test_stale_native_graph_rejects_replay_even_when_flash_passes(cpu_profiler):
    state = cpu_profiler
    state.stale_graph = "temporal"
    row = profiler.profile_case(state.case, state.args)
    assert row["status"] == "rejected_replay" and not row["exact_pass"]
    assert not row["mutation"]["temporal"]["output"]["passed"]
    assert row["baseline_mutation"]["passed"]
