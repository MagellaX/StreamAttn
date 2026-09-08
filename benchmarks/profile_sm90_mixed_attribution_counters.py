"""Matched Nsight replay of current paged affine code and actual FA2 execution."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

SCHEMA = "streamattn.sm90_mixed_attribution_counters.v1"


def collect(args):
    ncu = shutil.which("ncu")
    result = dict(schema=SCHEMA, complete=True, collected=False, rows=[],
        device=torch.cuda.get_device_name(), ncu=ncu,
        contract="kernel replay, cache-control none, clock-control none, three warmups; not latency evidence")
    if not ncu:
        result["status"] = "profiler_unavailable"
        return result
    result["ncu_version"] = subprocess.check_output([ncu, "--version"], text=True)
    args.build_dir.mkdir(parents=True, exist_ok=True)
    # D128/G8/BF16/HND, short, heterogeneous, and long-tail discovery cases.
    for case_index in (6, 8, 10):
        for target in ("natural_compact_interior/padded", "interior_kv_order/padded",
                       "interior_min2/padded", "flashinfer_fa2/packed"):
            label = f"c{case_index}_{target.replace('/', '_')}"
            csv_path, output = args.build_dir/(label+".csv"), args.build_dir/(label+".json")
            command = [ncu, "--clock-control", "none", "--cache-control", "none",
                "--replay-mode", "kernel", "--nvtx", "--nvtx-include", "streamattn_mixed_counter/",
                "--kernel-name-base", "demangled", "--csv", "--log-file", str(csv_path)]
            for section in ("LaunchStats", "SchedulerStats", "WarpStateStats", "InstructionStats",
                            "MemoryWorkloadAnalysis", "SpeedOfLight"):
                command += ["--section", section]
            command += [sys.executable, "-u", "benchmarks/profile_sm90_micro_prefill_mixed.py",
                "--suite", "causal", "--attribution", "--counter-target", target,
                "--case-index", str(case_index), "--cutlass-root", str(args.cutlass_root),
                "--build-dir", str(args.build_dir), "--output-json", str(output)]
            try:
                proc = subprocess.run(command, text=True, capture_output=True, timeout=600)
            except subprocess.TimeoutExpired:
                result.update(complete=False, status="timeout")
                return result
            raw = csv_path.read_text() if csv_path.exists() else ""
            result["rows"].append(dict(case_index=case_index, target=target, command=command,
                returncode=proc.returncode, profiler_csv=raw, stderr=proc.stderr,
                checked_result=json.loads(output.read_text()) if output.exists() else None))
            if "ERR_NVGPUCTRPERM" in raw+proc.stdout+proc.stderr:
                result["status"] = "counter_permission_denied"
                return result
            if proc.returncode or '"Metric Name"' not in raw:
                result.update(complete=False, status="counter_failed")
                result["rows"][-1]["stdout"] = proc.stdout
                return result
    result.update(collected=True, status="collected")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError("preserve existing evidence")
    result = collect(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2)+"\n")
    if not result["complete"]:
        raise RuntimeError("counter collection failed; evidence retained")


if __name__ == "__main__":
    main()
