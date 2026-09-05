"""Source-minimal natural R64 denominator-reduction ablation, not a public route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profile_sm90_micro_prefill import _capture, _elapsed_graph_ms  # noqa: E402
from benchmarks.profile_sm90_micro_prefill_semantics import (  # noqa: E402
    check_output, fp32_reference,
)
from stream_attention.backends.sm90.micro_prefill import NaturalMicroPrefillPlan  # noqa: E402
from stream_attention.backends.sm90.micro_prefill_semantics_sources import (  # noqa: E402
    CPP_SOURCE, _once, semantic_cuda_source,
)

SCHEMA = "streamattn.sm90_micro_prefill_deferred_sum.v1"


def deferred_source():
    source = semantic_cuda_source(128, "bf16", False)
    source = _once(source,
        "row_sum[row] += streamattn_quad_sum(local_sum);",
        "row_sum[row] += local_sum;")
    source = _once(source,
        "  CUTE_UNROLL\n  for (int row = 0; row < size<0>(output_rows); ++row) {",
        "  CUTE_UNROLL\n  for (int row = 0; row < kRowsPerThread; ++row) {\n"
        "    row_sum[row] = streamattn_quad_sum(row_sum[row]);\n  }\n"
        "  CUTE_UNROLL\n  for (int row = 0; row < size<0>(output_rows); ++row) {")
    return source


def compile_pair(args):
    from torch.utils.cpp_extension import load_inline

    extensions, sources = {}, {}
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        for name, source in (("control", semantic_cuda_source(128, "bf16", False)),
                             ("deferred", deferred_source())):
            identity = hashlib.sha256((CPP_SOURCE + source).encode()).hexdigest()
            directory = args.build_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            extensions[name] = load_inline(
                name="micro_sum_" + identity[:16], cpp_sources=CPP_SOURCE,
                cuda_sources=source, build_directory=str(directory),
                extra_include_paths=[str(args.cutlass_root / "include")],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math",
                                  "--expt-relaxed-constexpr", "--expt-extended-lambda",
                                  "--ptxas-options=-v,--warn-on-spills", "-lineinfo",
                                  "--keep", "--keep-dir=.",
                                  "-gencode=arch=compute_90a,code=sm_90a"],
                with_cuda=True, verbose=True,
            )
            sources[name] = identity
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous
    return extensions, sources


def profile_case(b, n, extensions):
    torch.manual_seed(7543 + n + b)
    q = torch.randn(b, 64, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, 2, n, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    po = torch.empty(b * 2 * 8, 16, 64, 128, device="cuda", dtype=torch.float32)
    pl = torch.empty(po.shape[:-1], device="cuda", dtype=torch.float32)
    empty = torch.empty(0, device="cuda", dtype=torch.int64)
    plans, graphs, results = {}, {}, {}
    for name, ext in extensions.items():
        out = torch.empty_like(q)
        # Use the retained plan's state shape and LSE reconstruction, with an
        # isolated combined launch. Both variants have identical host checks.
        plan = NaturalMicroPrefillPlan(
            query=q, key_cache=k, value_cache=v, output=out,
            partial_output=po.clone(), partial_lse=pl.clone(), num_splits=16,
            query_tiles=8, extension=ext, launch=ext.out, positions=(empty, empty),
        )
        plans[name] = plan
        graphs[name] = _capture(plan.run, warmup=3)
    for phase in ("initial", "mutated"):
        if phase == "mutated":
            q.normal_().mul_(3)
            k.normal_()
            v.normal_()
        ref, lse = fp32_reference(q, k, v)
        for name, plan in plans.items():
            plan.partial_output.fill_(float("nan"))
            plan.partial_lse.fill_(float("nan"))
            graphs[name].replay()
            torch.cuda.synchronize()
            results.setdefault(name, {})[phase] = check_output(plan, ref, lse)
    trials = []
    for trial in range(9):
        order = ("control", "deferred") if trial % 2 == 0 else ("deferred", "control")
        times = {name: 1000 * _elapsed_graph_ms(graphs[name], iterations=200) for name in order}
        trials.append(dict(times, speedup=times["control"] / times["deferred"]))
    return dict(batch=b, m=64, n=n, g=8, d=128, splits=16,
                correctness=results, paired_trials=trials,
                median_speedup=statistics.median(t["speedup"] for t in trials),
                passed=all(r[p]["passed"] for r in results.values() for p in ("initial", "mutated")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError("preserve existing evidence")
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(schema=SCHEMA, complete=False, provider="modal",
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(), rows=[],
                  ncu_path=shutil.which("ncu"), dynamic_counters_collected=False,
                  timing_contract="warm repeated CUDA graph; paired within one process; no external baseline")
    try:
        extensions, result["source_sha256"] = compile_pair(args)
        for b, n in ((1, 4096), (1, 16384), (2, 4096)):
            row = profile_case(b, n, extensions)
            result["rows"].append(row)
            if not row["passed"]:
                raise AssertionError("deferred sum failed correctness")
        result["complete"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
