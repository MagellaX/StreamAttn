from argparse import Namespace

from benchmarks.profile_gate0_projection_ncu import (
    _build_benchmark_command,
    _build_ncu_command,
    _compact_run,
)


def _args(**overrides):
    defaults = {
        "q_path": "q.pt",
        "k_path": "k.pt",
        "v_path": "",
        "dtype": "fp16",
        "head_index": 3,
        "projection_kind": "random",
        "projection_dim": 8,
        "projection_metadata_dtype": "fp16",
        "block_size": 16,
        "scan_region": "middle_only",
        "block_order": "recent_first",
        "sink_blocks": 2,
        "recent_blocks": 2,
        "filter_margin": 32.0,
        "error_budget": 1e-2,
        "blocks_per_program": 32,
        "words_per_program": 4,
        "seed": 0,
        "warmup": 3,
        "iters": 10,
        "profile_iters": 1,
        "ncu_bin": "ncu",
        "dry_run": True,
        "output_dir": "artifacts/ncu/test",
        "replay_mode": "kernel",
        "metrics": ["dram__bytes_read.sum"],
        "kernel_name": "regex:.*projection.*",
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_gate0_projection_ncu_benchmark_command_contains_backend_and_threshold():
    cmd = _build_benchmark_command(_args(), backend="triton_mask", threshold_mode="static")

    joined = " ".join(cmd)
    assert "benchmarks/profile_gate0_projection_kernel.py" in joined
    assert "--backend triton_mask" in joined
    assert "--threshold-mode static" in joined
    assert "--head-index 3" in joined


def test_gate0_projection_ncu_command_profiles_from_start_with_kernel_filter():
    cmd, output_base = _build_ncu_command(_args(), backend="triton_bitmask", threshold_mode="static")

    joined = " ".join(cmd)
    python_index = next(i for i, item in enumerate(cmd) if item.endswith("python") or item.endswith("python.exe"))
    kernel_name_index = cmd.index("--kernel-name")
    assert "--profile-from-start" not in joined
    assert "--cuda-profiler-api" not in joined
    assert "--kernel-name regex:.*projection.*" in joined
    assert kernel_name_index < python_index
    assert str(output_base).endswith("triton_bitmask_static")


def test_gate0_projection_ncu_compact_run_includes_kernel_and_metrics():
    run = {
        "backend": "triton_mask",
        "threshold_mode": "dynamic",
        "benchmark": {
            "candidate_scan_ms": 0.02,
            "scan_over_qk": 0.4,
            "q_projection_ms": 0.001,
            "estimated_speedup_vs_qk": 1.2,
            "actual_skip_recovery": 1.0,
            "false_skip_rate": 0.0,
            "middle_actual_skip_recovery": 1.0,
        },
        "benchmark_returncode": 0,
        "ncu_returncode": 11,
        "ncu_stdout_tail": "==ERROR== The application returned an error code (11).",
        "metrics": [
            {
                "metric": "dram__bytes_read.sum",
                "value": "1024",
                "unit": "byte",
            },
            {
                "metric": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
                "value": "70",
                "unit": "%",
            },
        ],
    }

    compact = _compact_run(run)

    assert compact["backend"] == "triton_mask"
    assert compact["threshold_mode"] == "dynamic"
    assert compact["benchmark_returncode"] == 0
    assert compact["ncu_returncode"] == 11
    assert compact["scan_over_qk"] == 0.4
    assert compact["dram_read_bytes"] == 1024.0
    assert compact["top_stall_metric"].endswith("long_scoreboard_per_warp_active.pct")
