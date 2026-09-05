"""Bounded Nsight attribution of the retained/deferred R64 denominator pair.

Counter replay is deliberately separate from uninstrumented paired timing.
An unavailable profiler or denied counters is recorded, never called evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profile_sm90_micro_prefill_deferred_sum import compile_pair
from benchmarks.profile_sm90_micro_prefill_semantics import check_output, fp32_reference
from stream_attention.backends.sm90.micro_prefill import NaturalMicroPrefillPlan

SCHEMA = "streamattn.sm90_micro_prefill_counters.v1"


def launch_one(args):
    torch.manual_seed(19043)
    torch.backends.cuda.matmul.allow_tf32 = False
    extensions, sources = compile_pair(args)
    b, n = args.batch, args.kv_len
    q = torch.randn(b, 64, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, 2, n, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    empty = torch.empty(0, device="cuda", dtype=torch.int64)
    ext = extensions[args.variant]
    plan = NaturalMicroPrefillPlan(
        query=q, key_cache=k, value_cache=v, output=torch.empty_like(q),
        partial_output=torch.empty(b*16, 16, 64, 128, device="cuda", dtype=torch.float32),
        partial_lse=torch.empty(b*16, 16, 64, device="cuda", dtype=torch.float32),
        num_splits=16, query_tiles=8, extension=ext, launch=ext.out,
        positions=(empty, empty),
    )
    for _ in range(3):
        plan.run()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("streamattn_counter")
    plan.run()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    expected, lse = fp32_reference(q, k, v)
    check = check_output(plan, expected, lse)
    print(json.dumps(dict(variant=args.variant, source_sha256=sources[args.variant], correctness=check)), flush=True)
    if not check["passed"]:
        raise AssertionError("counter target failed FP32 output/LSE check")


def collect(args):
    ncu = shutil.which("ncu")
    result = dict(schema=SCHEMA, complete=True, dynamic_counters_collected=False,
                  device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, ncu_path=ncu, rows=[],
                  timing_contract="instrumented attribution; unmanaged clocks; not speedup evidence")
    if not ncu:
        result["status"] = "profiler_unavailable"
        return result
    result["ncu_version"] = subprocess.check_output([ncu, "--version"], text=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    for b, n in ((1, 4096), (1, 16384), (2, 4096)):
        for variant in ("control", "deferred"):
            csv_path = args.build_dir / f"{variant}_b{b}_n{n}.csv"
            command = [
                ncu, "--clock-control", "none", "--nvtx", "--nvtx-include", "streamattn_counter/",
                "--kernel-name-base", "demangled", "--kernel-name",
                "regex:.*natural_wgmma_micro_prefill_partial_kernel.*",
                "--launch-count", "1", "--csv", "--log-file", str(csv_path),
            ]
            for section in ("LaunchStats", "SchedulerStats", "WarpStateStats", "InstructionStats", "MemoryWorkloadAnalysis", "SpeedOfLight"):
                command += ["--section", section]
            command += [
                sys.executable, "-u", str(Path(__file__).resolve()),
                "--variant", variant, "--batch", str(b), "--kv-len", str(n),
                "--cutlass-root", str(args.cutlass_root), "--build-dir", str(args.build_dir),
            ]
            try:
                proc = subprocess.run(command, text=True, capture_output=True, timeout=600)
            except subprocess.TimeoutExpired:
                result.update(complete=False, status="counter_probe_timeout")
                result["rows"].append(dict(batch=b, n=n, variant=variant, command=command))
                return result
            raw = csv_path.read_text() if csv_path.exists() else ""
            result["rows"].append(dict(batch=b, m=64, n=n, g=8, d=128, splits=16,
                                       variant=variant, command=command, returncode=proc.returncode,
                                       profiler_csv=raw, stdout=proc.stdout, stderr=proc.stderr))
            if "ERR_NVGPUCTRPERM" in raw + proc.stdout + proc.stderr:
                result["status"] = "counter_permission_denied"
                return result
            if (proc.returncode or '"Metric Name"' not in raw
                    or "streamattn_natural_wgmma_micro_prefill_partial_kernel" not in raw):
                result.update(complete=False, status="counter_probe_failed")
                return result
    result.update(dynamic_counters_collected=True, status="collected")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--variant", choices=("control", "deferred"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=4096)
    args = parser.parse_args()
    if args.variant:
        launch_one(args)
        return
    if args.output_json is None or args.output_json.exists():
        raise ValueError("specify a new output-json path")
    result = collect(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if not result["complete"]:
        raise RuntimeError("counter probe failed; evidence retained")


if __name__ == "__main__":
    main()
