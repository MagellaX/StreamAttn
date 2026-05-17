import pytest

from benchmarks.profile_gate1_ncu import (
    _compact_run,
    _parse_benchmark_output,
    _parse_ncu_csv,
    _summarize_metrics,
)


def test_ncu_csv_summary_maps_compact_fields():
    csv_text = """Metric Name,Metric Unit,Metric Value
sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,%,50.0
sm__pipe_tensor_active.avg.pct_of_peak_sustained_active,%,60.0
smsp__sass_thread_inst_executed_op_hmma_pred_on.sum,inst,"1,000"
smsp__sass_thread_inst_executed_op_hmma_pred_on.sum,inst,"2,000"
launch__registers_per_thread,register/thread,96
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,%,12.5
smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct,%,3.0
"""

    metrics = _parse_ncu_csv(csv_text)
    summary = _summarize_metrics(metrics)

    assert summary["tensor_active_pct"] == pytest.approx(55.0)
    assert summary["hmma_inst"] == pytest.approx(3000.0)
    assert summary["registers_per_thread"] == pytest.approx(96.0)
    assert summary["top_stall_metric"].endswith("long_scoreboard_per_warp_active.pct")
    assert summary["top_stall_value"] == pytest.approx(12.5)


def test_compact_run_includes_latency_active_fraction_and_metrics():
    run = {
        "force_mode": 0,
        "name": "normal",
        "benchmark": {
            "kernel_ms": 0.25,
            "stats": {
                "cta_tiles_total": 100,
                "cta_pv_executed": 20,
                "cta_pv_skipped": 80,
            },
        },
        "metrics": [
            {
                "metric": "sm__pipe_tensor_active.avg.pct_of_peak_sustained_active",
                "value": "42.0",
                "unit": "%",
            }
        ],
    }

    compact = _compact_run(run)

    assert compact["kernel_ms"] == pytest.approx(0.25)
    assert compact["active_pv_fraction"] == pytest.approx(0.2)
    assert compact["tensor_active_pct"] == pytest.approx(42.0)


def test_parse_benchmark_output_tolerates_prefix_noise():
    payload = _parse_benchmark_output('noise before\n{"kernel_ms": 0.1}\n')

    assert payload == {"kernel_ms": 0.1}
