import sys
from types import SimpleNamespace

import pytest
import torch

from benchmarks import sm90_mixed_attribution as attribution
from benchmarks.profile_sm90_micro_prefill_mixed import experiment_cases


def test_holdout_changes_lengths_and_batch_without_changing_variant_definitions():
    discovery, holdout = experiment_cases("causal"), experiment_cases("holdout")
    assert len(discovery) == len(holdout) == 24
    identity = lambda c: (tuple(c["query_lengths"]), tuple(c["kv_lengths"]))
    assert not set(map(identity, discovery)) & set(map(identity, holdout))
    assert {len(c["query_lengths"]) for c in holdout} == {5, 6}
    assert all(c["causal"] for c in holdout)


def test_poisoning_preserves_unused_split_identities():
    plan = SimpleNamespace(partial_output=torch.empty(3, 4, 64, 8),
        partial_lse=torch.empty(3, 4, 64), output=torch.empty(2, 8),
        tasks=torch.tensor([[0, 0, 0, 1], [0, 1, 1, 2], [2, 0, 0, 1]]))
    attribution.poison_live_states(plan)
    assert torch.isnan(plan.partial_lse[0, :2]).all()
    assert torch.isnan(plan.partial_lse[2, 0]).all()
    assert torch.isneginf(plan.partial_lse[1]).all()
    assert torch.isneginf(plan.partial_lse[0, 2:]).all()
    assert torch.isnan(plan.output).all()


def test_geometry_separates_reordering_from_repartitioning():
    c = next(c for c in experiment_cases("causal") if c["g"] == 8 and c["trace"] == "short")
    base = attribution.geometry(c, attribution.CONTROL)
    reordered = attribution.geometry(c, "interior_kv_order")
    widened = attribution.geometry(c, "interior_min2")
    assert base == reordered
    assert base["producer_ctas"] == 240 and widened["producer_ctas"] == 120
    assert min(widened["kv_tiles_per_task"]) == 2
    assert widened["merge_ctas"] == base["merge_ctas"]
    assert widened["partial_state_written_logical_bytes"] * 2 == base["partial_state_written_logical_bytes"]


@pytest.mark.parametrize("version,backend", [("0.6.14", "flashinfer_fa2"), ("0.6.13", "flashinfer_fa3")])
def test_telemetry_does_not_guess_unpinned_plan_layout(monkeypatch, version, backend):
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(__version__=version))
    fi = SimpleNamespace(wrapper=SimpleNamespace(_plan_info=list(range(15))))
    assert not attribution.flashinfer_telemetry(fi, backend)["decoded"]


def test_pinned_fa2_telemetry_decodes_byte_offsets_and_discards_padding(monkeypatch):
    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(__version__="0.6.13"))
    ws = torch.zeros(128, dtype=torch.uint8)
    for offset, values in ((0, [2, 0, 999]), (16, [1, 3, 999]), (32, [0, 1, 999]), (48, [128])):
        data = torch.tensor(values, dtype=torch.int32).view(torch.uint8)
        ws[offset:offset+len(data)] = data
    ws[64:67] = torch.tensor([1, 1, 0], dtype=torch.uint8)
    info = [3, 15, 0, 64, 0, 16, 32, 0, 0, 48, 0, 0, 64, 1, 1]
    fi = SimpleNamespace(wrapper=SimpleNamespace(_plan_info=info, _int_workspace_buffer=ws))
    result = attribution.flashinfer_telemetry(fi, "flashinfer_fa2")
    assert result["decoded"] and result["kv_chunk_tokens"] == 128
    assert result["valid_task_count"] == 2
    assert result["tasks"] == [[2, 1, 0], [0, 3, 1]]


@pytest.mark.parametrize("missing", [None, "baseline", "correctness", "binary", "pairs", "finite"])
def test_summary_fails_closed_and_keeps_internal_gain_distinct(missing):
    from benchmarks.summarize_sm90_post_affine_attribution import comparison

    times = {"natural_compact_interior/packed": 10, "interior_min2/packed": 5, "flashinfer_fa2/packed": 2}
    if missing == "finite":
        times["interior_min2/packed"] = float("nan")
    row = dict(case={}, passed=missing != "correctness",
        loaded_binary_provenance={n: dict(resolved=missing != "binary") for n in attribution.VARIANTS},
        graphs=dict(packed=dict(paired_trials=[] if missing == "pairs" else [dict(us=times)],
            fastest_tested_baseline=None if missing == "baseline" else dict(baseline_id="flashinfer_fa2", correctness_passed=True))))
    result = comparison(row, "packed", "interior_min2", "warm")
    if missing is not None:
        assert result is None
    else:
        assert result["control_speedup"] == 2
        assert result["baseline_speedup"] == 0.4
        assert result["control_all_pairs_win"] and not result["baseline_all_pairs_win"]
        assert comparison(row, "packed", "interior_min2", "perturbed") is None


def test_component_entry_point_does_not_change_default_out():
    from stream_attention.backends.sm90.micro_prefill_paged import PagedMicroPrefillPlan

    calls = []
    plan = SimpleNamespace(tasks=object(), output=object(), _arguments=lambda: ("args",),
        extension=SimpleNamespace(out=lambda *args: calls.append(args), components=lambda *args: calls.append(args)))
    assert PagedMicroPrefillPlan.run(plan) is plan.output
    PagedMicroPrefillPlan.run_component(plan, "producer")
    PagedMicroPrefillPlan.run_component(plan, "merge")
    assert calls == [("args",), ("args", 1), ("args", 2)]
    plan.tasks = None
    with pytest.raises(ValueError):
        PagedMicroPrefillPlan.run_component(plan, "producer")


def test_unavailable_counters_are_not_reported_as_collected(monkeypatch, tmp_path):
    from benchmarks import profile_sm90_mixed_attribution_counters as counters

    monkeypatch.setattr(counters.shutil, "which", lambda _: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test")
    result = counters.collect(SimpleNamespace(build_dir=tmp_path))
    assert result["complete"] and not result["collected"]
    assert result["status"] == "profiler_unavailable"


def test_counter_summary_requires_checked_launch_and_excludes_instrumented_time():
    from benchmarks.summarize_sm90_post_affine_attribution import summarize_counters

    raw = ('warning\n"ID","Process ID","Kernel Name","Metric Name","Metric Value","Metric Unit"\n'
           '"0","42","producer","Grid Size","240",""\n'
           '"0","42","producer","Duration","999","ns"\n')
    row = dict(case_index=6, target="test", returncode=0, profiler_csv=raw,
               checked_result=dict(complete=True, rows=[dict(passed=True)]))
    payload = dict(schema="streamattn.sm90_mixed_attribution_counters.v1",
                   complete=True, collected=True, rows=[row])
    result = summarize_counters(payload)
    assert result["collected"]
    assert result["rows"][0]["kernels"][0]["metrics"] == {"Grid Size": dict(value="240", unit="")}
    row["checked_result"]["rows"][0]["passed"] = False
    assert not summarize_counters(payload)["collected"]
