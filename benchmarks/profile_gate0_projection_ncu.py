"""Run Nsight Compute on Gate-0 projection scan kernels.

This profiles the standalone projection scan anatomy only. It does not build a
Gate-0 runtime or a Gate-1 inline filter.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate1_ncu import (
    DEFAULT_METRICS,
    LOCAL_MEMORY_METRICS,
    STALL_METRICS,
    _import_csv,
    _parse_benchmark_output,
    _summarize_metrics,
)


def _parse_str_values(raw: str):
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _kernel_name(backend: str, threshold_mode: str) -> str:
    return f"{backend}_{threshold_mode}"


def _append_kernel_args(cmd, args, *, backend: str, threshold_mode: str):
    cmd.extend(
        [
            "benchmarks/profile_gate0_projection_kernel.py",
            "--q-path",
            args.q_path,
            "--k-path",
            args.k_path,
            "--dtype",
            args.dtype,
            "--head-index",
            str(args.head_index),
            "--backend",
            backend,
            "--threshold-mode",
            threshold_mode,
            "--projection-kind",
            args.projection_kind,
            "--projection-dim",
            str(args.projection_dim),
            "--projection-metadata-dtype",
            args.projection_metadata_dtype,
            "--block-size",
            str(args.block_size),
            "--scan-region",
            args.scan_region,
            "--block-order",
            args.block_order,
            "--sink-blocks",
            str(args.sink_blocks),
            "--recent-blocks",
            str(args.recent_blocks),
            "--filter-margin",
            str(args.filter_margin),
            "--error-budget",
            str(args.error_budget),
            "--blocks-per-program",
            str(args.blocks_per_program),
            "--words-per-program",
            str(args.words_per_program),
            "--seed",
            str(args.seed),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--profile-iters",
            str(args.profile_iters),
        ]
    )
    if args.v_path:
        cmd.extend(["--v-path", args.v_path])
    return cmd


def _build_benchmark_command(args, *, backend: str, threshold_mode: str):
    return _append_kernel_args([sys.executable], args, backend=backend, threshold_mode=threshold_mode)


def _build_ncu_command(args, *, backend: str, threshold_mode: str):
    ncu = shutil.which(args.ncu_bin)
    if ncu is None:
        if args.dry_run:
            ncu = args.ncu_bin
        else:
            raise RuntimeError(f"could not find {args.ncu_bin!r} on PATH")
    output_base = Path(args.output_dir) / _kernel_name(backend, threshold_mode)
    cmd = [
        ncu,
        "--target-processes",
        "all",
        "--force-overwrite",
        "--export",
        str(output_base),
        "--replay-mode",
        args.replay_mode,
        "--metrics",
        ",".join(args.metrics),
    ]
    if args.kernel_name:
        cmd.extend(["--kernel-name", args.kernel_name])
    cmd.append(sys.executable)
    _append_kernel_args(cmd, args, backend=backend, threshold_mode=threshold_mode)
    return cmd, output_base


def _compact_run(run):
    benchmark = run.get("benchmark") or {}
    metrics = _summarize_metrics(run.get("metrics"))
    ncu_tail = run.get("ncu_stdout_tail") or ""
    return {
        "backend": run["backend"],
        "threshold_mode": run["threshold_mode"],
        "benchmark_returncode": run.get("benchmark_returncode"),
        "ncu_returncode": run.get("ncu_returncode"),
        "ncu_stdout_tail": ncu_tail[-1000:] if ncu_tail else None,
        "candidate_scan_ms": benchmark.get("candidate_scan_ms"),
        "scan_over_qk": benchmark.get("scan_over_qk"),
        "q_projection_ms": benchmark.get("q_projection_ms"),
        "estimated_speedup_vs_qk": benchmark.get("estimated_speedup_vs_qk"),
        "actual_skip_recovery": benchmark.get("actual_skip_recovery"),
        "false_skip_rate": benchmark.get("false_skip_rate"),
        "middle_actual_skip_recovery": benchmark.get("middle_actual_skip_recovery"),
        **{
            key: metrics.get(key)
            for key in [
                "dram_read_bytes",
                "l2_read_bytes",
                "registers_per_thread",
                "achieved_occupancy_pct",
                "eligible_warps_per_cycle",
                "local_memory_bytes",
                "top_stall_metric",
                "top_stall_value",
            ]
        },
        "metrics_missing": metrics.get("metrics_missing"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-path", required=True)
    parser.add_argument("--k-path", required=True)
    parser.add_argument("--v-path", default="")
    parser.add_argument("--backends", default="triton_mask,triton_bitmask")
    parser.add_argument("--threshold-modes", default="dynamic,static")
    parser.add_argument("--metric-preset", choices=["basic", "anatomy"], default="anatomy")
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--replay-mode", choices=["kernel", "application", "range"], default="kernel")
    parser.add_argument(
        "--kernel-name",
        default="regex:.*projection.*",
        help="Optional Nsight Compute kernel-name filter; pass an empty string to profile all kernels.",
    )
    parser.add_argument("--output-dir", default="artifacts/ncu/gate0_projection")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--head-index", type=int, default=-1)
    parser.add_argument("--projection-kind", choices=["random", "hadamard"], default="random")
    parser.add_argument("--projection-dim", type=int, default=8)
    parser.add_argument("--projection-metadata-dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--scan-region", choices=["all", "middle_only", "middle_plus_old"], default="middle_only")
    parser.add_argument("--block-order", choices=["sequential", "recent_first", "sink_recent_first"], default="recent_first")
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--filter-margin", type=float, default=32.0)
    parser.add_argument("--error-budget", type=float, default=1e-2)
    parser.add_argument("--blocks-per-program", type=int, default=32)
    parser.add_argument("--words-per-program", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--profile-iters", type=int, default=1)
    parser.add_argument("--continue-on-ncu-error", action="store_true")
    parser.add_argument("--summary-json-out", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.metrics is None:
        args.metrics = list(DEFAULT_METRICS)
        if args.metric_preset == "anatomy":
            args.metrics.extend(LOCAL_MEMORY_METRICS)
            args.metrics.extend(STALL_METRICS)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    runs = []
    for backend in _parse_str_values(args.backends):
        for threshold_mode in _parse_str_values(args.threshold_modes):
            if backend == "triton_score" and threshold_mode != "dynamic":
                continue
            benchmark_cmd = _build_benchmark_command(args, backend=backend, threshold_mode=threshold_mode)
            ncu_cmd, output_base = _build_ncu_command(args, backend=backend, threshold_mode=threshold_mode)
            run = {
                "backend": backend,
                "threshold_mode": threshold_mode,
                "output_base": str(output_base),
                "benchmark_command": benchmark_cmd,
                "command": ncu_cmd,
            }
            if not args.dry_run:
                benchmark_result = subprocess.run(
                    benchmark_cmd,
                    stderr=subprocess.STDOUT,
                    stdout=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                run["benchmark_returncode"] = benchmark_result.returncode
                run["benchmark_stdout_tail"] = benchmark_result.stdout[-8000:]
                if benchmark_result.returncode != 0:
                    if not args.continue_on_ncu_error:
                        raise subprocess.CalledProcessError(
                            benchmark_result.returncode,
                            benchmark_cmd,
                            output=benchmark_result.stdout,
                        )
                    runs.append(run)
                    continue
                run["benchmark"] = _parse_benchmark_output(benchmark_result.stdout)
                ncu_result = subprocess.run(
                    ncu_cmd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                run["ncu_returncode"] = ncu_result.returncode
                run["ncu_stdout_tail"] = ncu_result.stdout[-8000:]
                if ncu_result.returncode != 0:
                    if not args.continue_on_ncu_error:
                        raise subprocess.CalledProcessError(
                            ncu_result.returncode,
                            ncu_cmd,
                            output=ncu_result.stdout,
                        )
                else:
                    run["metrics"] = _import_csv(args, output_base)
            runs.append(run)

    payload = {"runs": runs}
    comparison = [_compact_run(run) for run in runs if "benchmark" in run or "metrics" in run]
    if comparison:
        payload["comparison"] = comparison
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.summary_json_out:
        Path(args.summary_json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json_out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
