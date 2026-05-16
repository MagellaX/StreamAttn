"""Run Nsight Compute on selected Gate-1 diagnostic modes.

This script is a thin wrapper around ``ncu`` and ``profile_gate1_kernel.py``.
It does not parse profiler output; it standardizes command construction so A100
and H100 runs compare the same force modes and shape.
"""

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_METRICS = [
    "sm__pipe_tensor_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum",
    "dram__bytes_read.sum",
    "lts__t_bytes_srcunit_tex_op_read.sum",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
]


def _mode_name(force_mode: int) -> str:
    return {
        0: "normal",
        5: "dense_equiv",
        7: "true_qk_scan",
        8: "qk_log_predicate_no_pv",
        9: "qk_exp_predicate_no_pv",
    }.get(force_mode, f"mode_{force_mode}")


def _build_command(args, force_mode: int):
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
    if args.causal:
        cmd.append("--causal")
    if args.precompute_bounds and args.kernel != "mass_specialized":
        cmd.append("--precompute-bounds")
    return cmd, output_base


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["generic", "mass_specialized"], default="generic")
    parser.add_argument("--modes", default="5,7,8,9,0")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
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
    parser.add_argument("--summary-json-out", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    runs = []
    for raw_mode in args.modes.split(","):
        force_mode = int(raw_mode.strip())
        cmd, output_base = _build_command(args, force_mode)
        runs.append(
            {
                "force_mode": force_mode,
                "name": _mode_name(force_mode),
                "output_base": str(output_base),
                "command": cmd,
            }
        )
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            if args.summary_json_out:
                run["metrics"] = _import_csv(args, output_base)
    if args.summary_json_out and not args.dry_run:
        Path(args.summary_json_out).write_text(
            json.dumps({"runs": runs}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"runs": runs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
