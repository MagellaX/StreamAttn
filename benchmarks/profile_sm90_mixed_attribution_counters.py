"""Matched Nsight replay of current paged affine code and actual FA2 execution."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

SCHEMA = "streamattn.sm90_mixed_attribution_counters.v1"
TRAFFIC_METRICS = (
    "dram__bytes_read.sum", "dram__bytes_write.sum",
    "lts__t_sectors_op_read.sum", "lts__t_sectors_op_write.sum",
    "smsp__inst_executed.sum",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
)


def collect(args):
    ncu = shutil.which("ncu")
    source = getattr(args, "source_correlated", False)
    producer_copy = getattr(args, "producer_copy", False)
    if producer_copy and not source:
        raise ValueError("producer-copy counters require source correlation")
    result = dict(schema=SCHEMA, complete=True, collected=False, rows=[],
        device=torch.cuda.get_device_name(), ncu=ncu,
        source_correlated=source,
        producer_copy=producer_copy,
        contract="kernel replay, cache-control none, clock-control none, three warmups; not latency evidence")
    if not ncu:
        result["status"] = "profiler_unavailable"
        return result
    result["ncu_version"] = subprocess.check_output([ncu, "--version"], text=True)
    if source:
        query = subprocess.run([ncu, "--query-metrics", "--query-metrics-mode", "all"],
                               text=True, capture_output=True, timeout=120)
        result["metric_query"] = dict(returncode=query.returncode, stderr=query.stderr)
        if query.returncode:
            result["metric_query"]["stdout"] = query.stdout
            result.update(complete=False, status="metric_query_failed")
            return result
        available = [m for m in TRAFFIC_METRICS if m in query.stdout]
        result["requested_metrics"] = list(TRAFFIC_METRICS)
        result["available_metrics"] = available
        result["unavailable_metrics"] = [m for m in TRAFFIC_METRICS if m not in available]
    args.build_dir.mkdir(parents=True, exist_ok=True)
    if source:
        # Keep cold compiler subprocesses outside Nsight injection. This run is
        # checked but never counted as profiling or timing evidence.
        warm_output = args.build_dir / "source_preflight.json"
        warm_command = [sys.executable, "-u", "benchmarks/profile_sm90_micro_prefill_mixed.py",
            "--suite", "causal", "--attribution", "--counter-target", "natural_compact_interior/padded",
            "--case-index", "6", "--cutlass-root", str(args.cutlass_root),
            "--build-dir", str(args.build_dir), "--output-json", str(warm_output)]
        if producer_copy:
            warm_command += ["--producer-copy"]
        print("prebuilding checked native and baseline modules outside Nsight", flush=True)
        warm = subprocess.run(warm_command, text=True, capture_output=True, timeout=900)
        checked = json.loads(warm_output.read_text()) if warm_output.exists() else {}
        result["preflight"] = dict(command=warm_command, returncode=warm.returncode,
            stdout=warm.stdout, stderr=warm.stderr, checked_result=checked)
        if (warm.returncode or not checked.get("complete")
                or not checked.get("rows", [{}])[0].get("loaded_binary_provenance", {}).get("flashinfer_fa2", {}).get("resolved")):
            result.update(complete=False, status="preflight_failed")
            return result
    # D128/G8/BF16/HND, short, heterogeneous, and long-tail discovery cases.
    for case_index in ((8,) if producer_copy else (6, 8, 10)):
        targets = ("natural_compact_interior/padded", "interior_q_vector/padded") if producer_copy else (
            ("natural_compact_interior/padded", "flashinfer_fa2/packed") if source else (
            "natural_compact_interior/padded", "interior_kv_order/padded",
            "interior_min2/padded", "flashinfer_fa2/packed"))
        for target in targets:
            print(f"collecting case {case_index}: {target}", flush=True)
            label = f"c{case_index}_{target.replace('/', '_')}"
            csv_path, output = args.build_dir/(label+".csv"), args.build_dir/(label+".json")
            command = [ncu, "--clock-control", "none", "--cache-control", "none",
                "--replay-mode", "kernel", "--nvtx", "--nvtx-include", "streamattn_mixed_counter/",
                "--kernel-name-base", "demangled", "--csv", "--log-file", str(csv_path)]
            for section in ("LaunchStats", "SchedulerStats", "WarpStateStats", "InstructionStats",
                            "MemoryWorkloadAnalysis", "SpeedOfLight"):
                command += ["--section", section]
            report = args.build_dir/(label+".ncu-rep")
            if source:
                symbol = ("natural_wgmma_micro_prefill_partial_kernel" if not target.startswith("flashinfer")
                          else "BatchPrefillWithPagedKVCacheKernel")
                command += ["--kernel-name", "regex:.*"+symbol+".*", "--launch-count", "1",
                    "--section", "SourceCounters", "--import-source", "yes",
                    "--source-folders", str(args.build_dir), "--export", str(report), "--page", "raw"]
                if available:
                    command += ["--metrics", ",".join(available)]
            command += [sys.executable, "-u", "benchmarks/profile_sm90_micro_prefill_mixed.py",
                "--suite", "causal", "--attribution", "--counter-target", target,
                "--case-index", str(case_index), "--cutlass-root", str(args.cutlass_root),
                "--build-dir", str(args.build_dir), "--output-json", str(output)]
            if producer_copy:
                command += ["--producer-copy"]
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
            header = '"Kernel Name"' if source else '"Metric Name"'
            if proc.returncode or header not in raw:
                result.update(complete=False, status="counter_failed")
                result["rows"][-1]["stdout"] = proc.stdout
                return result
            if source:
                export_command = [ncu, "--import", str(report), "--page", "source",
                                  "--print-source", "cuda,sass", "--csv"]
                export = subprocess.run(export_command, text=True, capture_output=True, timeout=120)
                row = result["rows"][-1]
                row.update(source_export_command=export_command, source_csv=export.stdout,
                           source_export_stderr=export.stderr, source_export_returncode=export.returncode)
                # Preserve exact generated source and its hash, not inferred source line numbers.
                row["generated_sources"] = {str(p): dict(text=p.read_text(),
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in args.build_dir.rglob("cuda.cu")}
                if export.returncode or not export.stdout.strip():
                    result.update(complete=False, status="source_export_failed")
                    return result
    result.update(collected=True, status="collected")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-correlated", action="store_true")
    parser.add_argument("--producer-copy", action="store_true")
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
