"""Probe the decode-backend integration surface.

This is a backend reconnaissance tool, not a performance benchmark.  It records
what the installed FlashInfer package exposes, whether custom/JIT hooks are
visible from Python, whether source/header files are present in the wheel, and
optionally whether this environment can build and run a minimal CUDA extension.

The output is meant to guide the next implementation choice:

* patch/extend a FlashInfer-style JIT/backend path;
* build a standalone CUDA/CuTe backend with the same plan/run shape;
* keep Triton only for science kernels.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import torch


INTERESTING_SYMBOLS = [
    "flashinfer.decode.single_decode_with_kv_cache",
    "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper",
    "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.__init__",
    "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.plan",
    "flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.run",
    "flashinfer.decode.get_single_decode_module",
    "flashinfer.decode.get_batch_decode_module",
    "flashinfer.decode.get_batch_decode_jit_module",
    "flashinfer.jit.gen_single_decode_module",
    "flashinfer.jit.gen_batch_decode_module",
    "flashinfer.jit.gen_customize_batch_decode_module",
    "flashinfer.jit.gen_customize_batch_prefill_module",
    "flashinfer.prefill.get_single_prefill_module",
]


PATTERNS = [
    "use_tensor_cores",
    "get_single_prefill_module",
    "get_single_decode_module",
    "BatchDecodeWithPagedKVCacheWrapper",
    "jit_args",
    "gen_customize_batch_decode_module",
    "gen_customize_batch_prefill_module",
    "backend",
    "plan_info_vec",
    "TensorLayout",
]


def _safe_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, repr(exc)


def _dist_info(name: str) -> dict[str, Any]:
    try:
        dist = importlib.metadata.distribution(name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}
    files = list(dist.files or [])
    suffix_counts: dict[str, int] = {}
    interesting = []
    for file in files:
        text = str(file)
        suffix = Path(text).suffix or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        if any(token in text for token in ("decode", "prefill", "jit", "include", "csrc", "cubin")):
            interesting.append(text)
    return {
        "available": True,
        "version": dist.version,
        "file_count": len(files),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "interesting_files_sample": interesting[:120],
    }


def _resolve_symbol(path: str):
    parts = path.split(".")
    module = None
    for idx in range(len(parts), 0, -1):
        module_name = ".".join(parts[:idx])
        try:
            module = importlib.import_module(module_name)
            obj = module
            remainder = parts[idx:]
            break
        except Exception:
            continue
    else:
        raise ImportError(path)
    for part in remainder:
        obj = getattr(obj, part)
    return obj


def _source_summary(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["signature"] = str(inspect.signature(obj))
    except Exception as exc:
        out["signature_error"] = repr(exc)
    try:
        source = inspect.getsource(obj)
        lines = source.splitlines()
        out["source_file"] = inspect.getsourcefile(obj)
        out["source_line_count"] = len(lines)
        out["source_first_lines"] = lines[:30]
        out["pattern_hits"] = {
            pattern: [idx + 1 for idx, line in enumerate(lines) if pattern in line][:12]
            for pattern in PATTERNS
            if any(pattern in line for line in lines)
        }
    except Exception as exc:
        out["source_error"] = repr(exc)
    return out


def _symbol_summaries() -> dict[str, Any]:
    summaries = {}
    for symbol in INTERESTING_SYMBOLS:
        try:
            summaries[symbol] = _source_summary(_resolve_symbol(symbol))
        except Exception as exc:
            summaries[symbol] = {"error": repr(exc)}
    return summaries


def _walk_package_files(package_root: Path, *, limit: int = 200) -> dict[str, Any]:
    suffix_counts: dict[str, int] = {}
    interesting = []
    source_like = []
    if not package_root.exists():
        return {"exists": False}
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(package_root)).replace("\\", "/")
        suffix = path.suffix or "<none>"
        suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        if any(token in rel for token in ("decode", "prefill", "jit", "include", "csrc", "cubin")):
            interesting.append(rel)
        if suffix in {".cu", ".cuh", ".h", ".hpp", ".cc", ".cpp", ".so"}:
            source_like.append(rel)
    return {
        "exists": True,
        "root": str(package_root),
        "suffix_counts": dict(sorted(suffix_counts.items())),
        "interesting_files_sample": interesting[:limit],
        "source_like_files_sample": source_like[:limit],
        "has_cuda_sources": any(Path(item).suffix in {".cu", ".cuh"} for item in source_like),
        "has_shared_objects": any(Path(item).suffix == ".so" for item in source_like),
    }


def _grep_files(paths: Iterable[Path], *, root: Path, patterns: list[str], limit: int = 12) -> dict[str, Any]:
    hits: dict[str, list[dict[str, Any]]] = {pattern: [] for pattern in patterns}
    for path in paths:
        if not path.is_file() or path.suffix not in {".py", ".cu", ".cuh", ".h", ".hpp", ".cc", ".cpp"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern in patterns:
            if len(hits[pattern]) >= limit:
                continue
            match = re.search(re.escape(pattern), text)
            if match:
                line_no = text[: match.start()].count("\n") + 1
                hits[pattern].append(
                    {
                        "path": str(path.relative_to(root)).replace("\\", "/"),
                        "line": line_no,
                    }
                )
    return {pattern: rows for pattern, rows in hits.items() if rows}


def _compile_cuda_smoke() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "reason": "CUDA is not available"}
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as exc:
        return {"available": False, "reason": f"cpp_extension import failed: {exc!r}"}

    cpp_source = r"""
#include <torch/extension.h>

void streamattn_add_one_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &streamattn_add_one_cuda, "StreamAttn backend compile smoke");
}
"""
    cuda_source = r"""
#include <torch/extension.h>

__global__ void add_one_kernel(float* x, int64_t n) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    x[idx] += 1.0f;
  }
}

void streamattn_add_one_cuda(torch::Tensor x) {
  const int threads = 256;
  const int64_t n = x.numel();
  const int blocks = (n + threads - 1) / threads;
  add_one_kernel<<<blocks, threads>>>(x.data_ptr<float>(), n);
}
"""
    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="streamattn_cuda_smoke_") as tmp:
            module = load_inline(
                name="streamattn_cuda_smoke",
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                build_directory=tmp,
                verbose=False,
                with_cuda=True,
                extra_cuda_cflags=["-O2"],
            )
            compile_ms = (time.perf_counter() - start) * 1000.0
            x = torch.zeros(16, device="cuda", dtype=torch.float32)
            module.add_one(x)
            torch.cuda.synchronize()
            max_value = float(x.max().item())
        return {
            "available": True,
            "compile_ms": compile_ms,
            "run_max_value": max_value,
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        }
    except Exception as exc:  # pragma: no cover - compiler/environment dependent
        return {
            "available": False,
            "reason": repr(exc),
            "elapsed_ms": (time.perf_counter() - start) * 1000.0,
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        }
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


def _recommendation(result: dict[str, Any]) -> dict[str, Any]:
    flashinfer_pkg = result.get("packages", {}).get("flashinfer_package", {})
    symbols = result.get("symbols", {})
    wrapper_init = symbols.get("flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper.__init__", {})
    single_decode = symbols.get("flashinfer.decode.single_decode_with_kv_cache", {})
    compile_smoke = result.get("cuda_compile_smoke", {})
    package_walk = result.get("package_walk", {}).get("flashinfer", {})

    has_jit_args = "jit_args" in str(wrapper_init.get("signature", "")) or bool(
        (wrapper_init.get("pattern_hits") or {}).get("jit_args")
    )
    tc_uses_prefill = bool((single_decode.get("pattern_hits") or {}).get("get_single_prefill_module"))
    has_sources = bool(package_walk.get("has_cuda_sources"))
    can_compile = bool(compile_smoke.get("available"))

    if has_jit_args and can_compile:
        path = "flashinfer_style_custom_jit_or_cuda_extension"
    elif can_compile:
        path = "custom_cuda_cute_head_mode_backend"
    else:
        path = "external_backend_required"

    return {
        "recommended_next_path": path,
        "facts": {
            "flashinfer_dist_available": bool(flashinfer_pkg.get("available")),
            "batch_decode_exposes_jit_args": has_jit_args,
            "single_decode_tc_uses_prefill_path": tc_uses_prefill,
            "installed_wheel_has_cuda_sources": has_sources,
            "cuda_extension_compile_smoke_passed": can_compile,
        },
        "interpretation": (
            "The StreamAttn head-mode path should attach at a plan/run decode boundary. "
            "If the installed FlashInfer wheel lacks editable CUDA sources, use its API "
            "surface as the model but build StreamAttn's own CUDA/CuTe backend."
        ),
    }


def probe(args: argparse.Namespace) -> dict[str, Any]:
    flashinfer, flashinfer_error = _safe_import("flashinfer")
    flashinfer_decode, decode_error = _safe_import("flashinfer.decode")
    flashinfer_jit, jit_error = _safe_import("flashinfer.jit")

    package_roots = {}
    grep_hits = {}
    if flashinfer is not None and getattr(flashinfer, "__file__", None):
        root = Path(flashinfer.__file__).resolve().parent
        package_roots["flashinfer"] = _walk_package_files(root)
        grep_hits["flashinfer"] = _grep_files(root.rglob("*"), root=root, patterns=PATTERNS)

    result = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        },
        "imports": {
            "flashinfer": {
                "ok": flashinfer is not None,
                "error": flashinfer_error,
                "file": getattr(flashinfer, "__file__", None) if flashinfer is not None else None,
                "version": getattr(flashinfer, "__version__", None) if flashinfer is not None else None,
            },
            "flashinfer.decode": {"ok": flashinfer_decode is not None, "error": decode_error},
            "flashinfer.jit": {"ok": flashinfer_jit is not None, "error": jit_error},
        },
        "packages": {
            "flashinfer_package": _dist_info("flashinfer-python"),
            "flashinfer_cubin": _dist_info("flashinfer-cubin"),
        },
        "package_walk": package_roots,
        "grep_hits": grep_hits,
        "symbols": _symbol_summaries() if flashinfer is not None else {},
    }
    if args.compile_smoke:
        result["cuda_compile_smoke"] = _compile_cuda_smoke()
    else:
        result["cuda_compile_smoke"] = {"available": None, "reason": "not requested"}
    result["recommendation"] = _recommendation(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-smoke", action="store_true")
    parser.add_argument("--summary-json-out", default="")
    args = parser.parse_args()
    result = probe(args)
    text = json.dumps(result, indent=2, sort_keys=True, default=str)
    if args.summary_json_out:
        path = Path(args.summary_json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
