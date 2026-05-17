"""Run Nsight Compute on selected Gate-1 diagnostic modes.

The wrapper keeps two measurements separate:

* a clean CUDA-event benchmark from ``profile_gate1_kernel.py`` for latency;
* an Nsight Compute report import for hardware counters.

The final JSON includes a compact per-mode comparison so A100/H100 runs can be
read without manually opening every ``.ncu-rep`` file.
"""

import argparse
import csv
import io
import json
import statistics
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_METRICS = [
    "sm__pipe_tensor_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum",
    "dram__bytes_read.sum",
    "lts__t_bytes_srcunit_tex_op_read.sum",
    "launch__registers_per_thread",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
]

STALL_METRICS = [
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
]

LOCAL_MEMORY_METRICS = [
    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
]

METRIC_ALIASES = {
    "tensor_active_pct": "sm__pipe_tensor_active.avg.pct_of_peak_sustained_active",
    "hmma_inst": "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum",
    "dram_read_bytes": "dram__bytes_read.sum",
    "l2_read_bytes": "lts__t_bytes_srcunit_tex_op_read.sum",
    "registers_per_thread": "launch__registers_per_thread",
    "branch_uniform_pct": "smsp__sass_average_branch_targets_threads_uniform.pct",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "eligible_warps_per_cycle": "smsp__warps_eligible.avg.per_cycle_active",
    "local_load_bytes": "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",
    "local_store_bytes": "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",
}


def _mode_name(force_mode: int) -> str:
    return {
        0: "normal",
        5: "dense_equiv",
        7: "true_qk_scan",
        8: "qk_log_predicate_no_pv",
        9: "qk_exp_predicate_no_pv",
    }.get(force_mode, f"mode_{force_mode}")


def _append_benchmark_args(cmd, args, force_mode: int, *, return_stats: bool = False):
    cmd.extend(
        [
            "benchmarks/profile_gate1_kernel.py",
            "--kernel",
            args.kernel,
            "--batch",
            str(args.batch),
            "--seq-q",
            str(args.seq_q),
            "--seq-k",
            str(args.seq_k),
            "--heads",
            str(args.heads),
            "--dim",
            str(args.dim),
            "--dtype",
            args.dtype,
            "--pattern",
            args.pattern,
            "--active-fraction",
            str(args.active_fraction),
            "--block-size",
            str(args.block_size),
            "--tile-size-q",
            str(args.tile_size_q),
            "--force-mode",
            str(force_mode),
            "--skip-predicate",
            args.skip_predicate,
            "--num-warps",
            str(args.num_warps),
            "--num-stages",
            str(args.num_stages),
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ]
    )
    if args.causal:
        cmd.append("--causal")
    if args.precompute_bounds and args.kernel != "mass_specialized":
        cmd.append("--precompute-bounds")
    if return_stats:
        cmd.append("--return-stats")
    return cmd


def _build_benchmark_command(args, force_mode: int):
    return _append_benchmark_args(
        [sys.executable],
        args,
        force_mode,
        return_stats=args.collect_stats,
    )


def _build_ncu_command(args, force_mode: int):
    ncu = shutil.which(args.ncu_bin)
    if ncu is None:
        if args.dry_run:
            ncu = args.ncu_bin
        else:
            raise RuntimeError(f"could not find {args.ncu_bin!r} on PATH")
    output_base = Path(args.output_dir) / f"{args.kernel}_{_mode_name(force_mode)}"
    cmd = [
        ncu,
        "--target-processes",
        "all",
        "--force-overwrite",
        "--export",
        str(output_base),
        "--metrics",
        ",".join(args.metrics),
        sys.executable,
    ]
    return _append_benchmark_args(cmd, args, force_mode), output_base


def _parse_ncu_csv(text: str):
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        metric = row.get("Metric Name") or row.get("Metric")
        value = row.get("Metric Value") or row.get("Value")
        unit = row.get("Metric Unit") or row.get("Unit")
        if metric:
            rows.append({"metric": metric, "value": value, "unit": unit})
    return rows


def _import_csv(args, output_base: Path):
    report = output_base.with_suffix(".ncu-rep")
    cmd = [args.ncu_bin, "--import", str(report), "--csv"]
    output = subprocess.check_output(cmd, text=True)
    return _parse_ncu_csv(output)


def _parse_number(raw):
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace(",", "")
    if not text or text.upper() in {"N/A", "NA", "NAN"}:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _parse_benchmark_output(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[start:])
                return payload
            except json.JSONDecodeError:
                continue
        raise


def _metric_values(metrics):
    values = {}
    for row in metrics or []:
        metric = row.get("metric")
        value = _parse_number(row.get("value"))
        if metric is not None and value is not None:
            values.setdefault(metric, []).append(value)
    return values


def _aggregate_metric(metric: str, values_by_metric):
    values = values_by_metric.get(metric)
    if not values:
        return None
    if metric.endswith(".sum"):
        return float(sum(values))
    return float(statistics.median(values))


def _summarize_metrics(metrics):
    values_by_metric = _metric_values(metrics)
    compact = {}
    for field, metric in METRIC_ALIASES.items():
        compact[field] = _aggregate_metric(metric, values_by_metric)

    local_bytes = [
        compact.get("local_load_bytes"),
        compact.get("local_store_bytes"),
    ]
    compact["local_memory_bytes"] = (
        float(sum(value for value in local_bytes if value is not None))
        if any(value is not None for value in local_bytes)
        else None
    )

    stall_candidates = []
    for metric, values in values_by_metric.items():
        if "stalled" not in metric and "stall" not in metric:
            continue
        value = _aggregate_metric(metric, values_by_metric)
        if value is not None:
            stall_candidates.append((metric, value))
    if stall_candidates:
        metric, value = max(stall_candidates, key=lambda item: item[1])
        compact["top_stall_metric"] = metric
        compact["top_stall_value"] = value
    else:
        compact["top_stall_metric"] = None
        compact["top_stall_value"] = None

    compact["metrics_available"] = sorted(values_by_metric)
    compact["metrics_missing"] = [
        metric
        for metric in METRIC_ALIASES.values()
        if metric not in values_by_metric
    ]
    return compact


def _compact_run(run):
    benchmark = run.get("benchmark") or {}
    metrics = _summarize_metrics(run.get("metrics"))
    stats = benchmark.get("stats") or {}
    cta_total = stats.get("cta_tiles_total") or 0
    cta_executed = stats.get("cta_pv_executed")
    active_frac = (
        cta_executed / cta_total
        if cta_total and cta_executed is not None
        else None
    )
    return {
        "force_mode": run["force_mode"],
        "name": run["name"],
        "kernel_ms": benchmark.get("kernel_ms"),
        "active_pv_fraction": active_frac,
        "cta_pv_executed": cta_executed,
        "cta_pv_skipped": stats.get("cta_pv_skipped"),
        "cta_tiles_total": cta_total or None,
        **{
            key: metrics.get(key)
            for key in [
                "tensor_active_pct",
                "hmma_inst",
                "dram_read_bytes",
                "l2_read_bytes",
                "registers_per_thread",
                "branch_uniform_pct",
                "achieved_occupancy_pct",
                "eligible_warps_per_cycle",
                "local_memory_bytes",
                "top_stall_metric",
                "top_stall_value",
            ]
        },
        "metrics_missing": metrics.get("metrics_missing"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["generic", "mass_specialized"], default="generic")
    parser.add_argument("--modes", default="5,7,8,9,0")
    parser.add_argument("--metric-preset", choices=["basic", "anatomy"], default="basic")
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--output-dir", default="artifacts/ncu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-q", type=int, default=4096)
    parser.add_argument("--seq-k", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--pattern", choices=["random", "peaked"], default="peaked")
    parser.add_argument("--active-fraction", type=float, default=0.0625)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--tile-size-q", type=int, default=64)
    parser.add_argument("--skip-predicate", choices=["mass", "value_bound"], default="mass")
    parser.add_argument("--precompute-bounds", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--collect-stats", action="store_true")
    parser.add_argument("--summary-json-out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.metrics is None:
        args.metrics = list(DEFAULT_METRICS)
        if args.metric_preset == "anatomy":
            args.metrics.extend(LOCAL_MEMORY_METRICS)
            args.metrics.extend(STALL_METRICS)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    runs = []
    for raw_mode in args.modes.split(","):
        force_mode = int(raw_mode.strip())
        ncu_cmd, output_base = _build_ncu_command(args, force_mode)
        benchmark_cmd = _build_benchmark_command(args, force_mode)
        run = {
            "force_mode": force_mode,
            "name": _mode_name(force_mode),
            "output_base": str(output_base),
            "benchmark_command": benchmark_cmd,
            "command": ncu_cmd,
        }
        if not args.dry_run:
            benchmark_output = subprocess.check_output(
                benchmark_cmd,
                stderr=subprocess.STDOUT,
                text=True,
            )
            run["benchmark"] = _parse_benchmark_output(benchmark_output)
            subprocess.run(ncu_cmd, check=True)
            if args.summary_json_out:
                run["metrics"] = _import_csv(args, output_base)
        runs.append(run)

    comparison = [_compact_run(run) for run in runs if "benchmark" in run or "metrics" in run]
    payload = {"runs": runs}
    if comparison:
        payload["comparison"] = comparison
    if args.summary_json_out and not args.dry_run:
        Path(args.summary_json_out).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
