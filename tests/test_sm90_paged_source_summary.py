import csv
import hashlib
import io
import json

import pytest

from benchmarks.profile_sm90_micro_prefill_mixed import experiment_cases
from benchmarks.sm90_mixed_attribution import CONTROL, geometry, useful_work
from benchmarks.summarize_sm90_paged_source import flashinfer_work, number, raw_metrics, regions, summarize
from stream_attention.backends.sm90.micro_prefill_ragged_sources import ragged_cuda_source


def capture():
    source = ragged_cuda_source(128, "bf16", True, "interior")
    line = next(start for name, start, _ in regions(source) if name == "q_staging")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["File Path", "/tmp/cuda.cu"])
    writer.writerow(["Function Name", "producer"])
    writer.writerow(["Line No", "Source", "Address", "Source", "# Samples", "stall_long_sb"])
    writer.writerow([str(line), "query staging", "-", "-", "10", "7"])
    writer.writerow(["", "", "0x123", "ST.E.U16 R0, R1", "10", "7"])
    raw = ('warning\n"ID","Kernel Name","smsp__pcsamp_sample_count",'
           '"smsp__pcsamp_warps_issue_stalled_long_scoreboard","dram__bytes_read.sum"\n'
           '"","","sample","sample","byte"\n"0","producer","10","7","1,024"\n')
    return dict(case_index=6, target="natural_compact_interior/padded", returncode=0,
                source_export_returncode=0, source_csv=output.getvalue(), profiler_csv=raw,
                generated_sources={"/tmp/cuda.cu": dict(text=source, sha256=hashlib.sha256(source.encode()).hexdigest())},
                checked_result=dict(complete=True, rows=[dict(passed=True, case=experiment_cases("causal")[6])]))


def payload(row):
    return dict(schema="streamattn.sm90_mixed_attribution_counters.v1", source_correlated=True,
                complete=True, collected=True, rows=[row], available_metrics=["dram__bytes_read.sum"])


def test_normalizes_traffic_and_maps_exact_captured_source():
    captured = capture()
    checked = captured["checked_result"]["rows"][0]
    checked["counter_work"] = json.loads(json.dumps(dict(useful=useful_work(checked["case"]),
                                                        native=geometry(checked["case"], CONTROL))))
    result = summarize(payload(captured))
    assert not result["complete"]  # One launch does not complete the six-launch matrix.
    row = result["rows"][0]
    assert row["regions"] == {"q_staging": {"# Samples": 10, "stall_long_sb": 7}}
    assert row["counters_per_visible_head_pair"]["dram__bytes_read.sum"] == 1024 / row["useful_work"]["visible_head_pairs"]
    assert not row["attributed_stalls_available"]


def test_missing_external_lineinfo_is_explicit_not_a_native_source_failure():
    row = capture()
    row.update(target="flashinfer_fa2/packed", source_csv="No lineinfo available")
    summary = summarize(payload(row))["rows"][0]
    assert not summary["source_correlation_available"]
    assert summary["aggregate_totals_match"] is None
    row["target"] = "natural_compact_interior/padded"
    with pytest.raises(ValueError, match="no source"):
        summarize(payload(row))


def test_fa2_work_uses_actual_tile_traits_and_live_plan():
    case = dict(query_lengths=[3], kv_lengths=[65], g=8, hq=16, d=128, causal=True)
    telemetry = dict(version="0.6.13", decoded=True, plan=dict(cta_tile_q=128, split_kv=1),
                     tasks=[[0, 0, 0], [0, 0, 1]], kv_chunk_tokens=64)
    kernel = "BatchPrefillWithPagedKVCacheKernel<KernelTraits<1, 128, 2, 4, 8, 8, 4, 1, 0, __nv_bfloat16>>"
    result = flashinfer_work(case, telemetry, kernel)
    assert result["iterations_per_live_task"] == [1, 1]
    assert result["scheduled_head_pairs"] == 2*128*64*2
    assert not flashinfer_work(case, telemetry, kernel.replace("<1,", "<0,"))["decoded"]
    telemetry["version"] = "unknown"
    assert not flashinfer_work(case, telemetry, kernel)["decoded"]


@pytest.mark.parametrize("fault", ["hash", "samples", "unchecked", "duplicate"])
def test_rejects_unverified_evidence(fault):
    row = capture()
    data = payload(row)
    if fault == "hash":
        row["generated_sources"]["/tmp/cuda.cu"]["sha256"] = "wrong"
    elif fault == "samples":
        row["profiler_csv"] = row["profiler_csv"].replace('"10"', '"11"')
    elif fault == "unchecked":
        row["checked_result"]["rows"][0]["passed"] = False
    else:
        data["rows"].append(row)
    with pytest.raises(ValueError):
        summarize(data)


def test_missing_counters_are_not_zero_and_multiple_launches_are_rejected():
    assert number({}, "metric") is None
    with pytest.raises(ValueError, match="non-finite"):
        number({"metric": "nan"}, "metric")
    with pytest.raises(ValueError, match="one filtered"):
        raw_metrics(capture()["profiler_csv"] + '"1","producer","10","7","1,024"\n')


def test_pc_instruction_totals_must_match_kernel_counter():
    row = capture()
    fields = list(csv.reader(io.StringIO(row["source_csv"])))
    for entry in fields[2:]:
        entry.append("Instructions Executed" if entry[0] == "Line No" else "32")
    output = io.StringIO()
    csv.writer(output).writerows(fields)
    row["source_csv"] = output.getvalue()
    row["profiler_csv"] = row["profiler_csv"].replace(
        '"dram__bytes_read.sum"', '"smsp__inst_executed.sum"').replace('"1,024"', '"32"')
    data = payload(row)
    data["available_metrics"] = ["smsp__inst_executed.sum"]
    assert summarize(data)["rows"][0]["pc_instruction_total"] == 32
    row["profiler_csv"] = row["profiler_csv"].replace('"32"', '"33"')
    with pytest.raises(ValueError, match="PC instructions"):
        summarize(data)
