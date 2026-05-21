"""Probe tensor-core backend paths for StreamAttn head-mode decode.

The scalar CUDA head-mode prototypes proved the scheduler branch is correct,
but also proved that a scalar exact branch cannot compete with FlashInfer.  This
probe answers the next engineering question: which tensor-core backend route is
actually actionable in the current environment?

It deliberately does not benchmark attention.  Instead it checks whether the
environment can compile a tiny CuTe/CUTLASS or ThunderKittens CUDA translation
unit, records the relevant source/header locations, and emits a blunt next-path
recommendation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]

CUTLASS_REPO = "https://github.com/NVIDIA/cutlass.git"
THUNDERKITTENS_REPO = "https://github.com/HazyResearch/ThunderKittens.git"


def _run(
    cmd: list[str],
    *,
    cwd: Optional[Path] = None,
    timeout: int = 600,
) -> dict[str, Any]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout,
    )
    return {
        "cmd": cmd,
        "returncode": result.returncode,
        "output_tail": result.stdout[-12000:],
    }


def _which(name: str) -> Optional[str]:
    path = shutil.which(name)
    return str(Path(path).resolve()) if path else None


def _version_cmd(cmd: list[str]) -> dict[str, Any]:
    if not _which(cmd[0]):
        return {"available": False, "cmd": cmd}
    result = _run(cmd, timeout=60)
    return {
        "available": result["returncode"] == 0,
        "cmd": cmd,
        "returncode": result["returncode"],
        "output_tail": result["output_tail"],
    }


def _env_paths(names: Iterable[str]) -> list[Path]:
    roots: list[Path] = []
    for name in names:
        value = os.environ.get(name)
        if not value:
            continue
        for part in value.split(os.pathsep):
            if part:
                roots.append(Path(part).expanduser())
    return roots


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        key = str(resolved).lower() if os.name == "nt" else str(resolved)
        if key in seen:
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _clone_repo(url: str, dest: Path) -> dict[str, Any]:
    if dest.exists():
        return {"requested": True, "available": True, "path": str(dest), "reason": "already_exists"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git",
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        url,
        str(dest),
    ]
    result = _run(cmd, timeout=1800)
    return {
        "requested": True,
        "available": result["returncode"] == 0 and dest.exists(),
        "path": str(dest),
        "returncode": result["returncode"],
        "output_tail": result["output_tail"],
    }


def _candidate_roots(args: argparse.Namespace) -> dict[str, list[Path]]:
    third_party = REPO_ROOT / "third_party"
    artifact_sources = Path(args.checkout_dir).expanduser()
    roots = {
        "cutlass": _env_paths(["CUTLASS_PATH", "CUTLASS_ROOT", "CUTE_PATH", "CUTE_ROOT"])
        + [
            third_party / "cutlass",
            artifact_sources / "cutlass",
        ],
        "thunderkittens": _env_paths(
            ["THUNDERKITTENS_PATH", "THUNDERKITTENS_ROOT", "TK_PATH", "TK_ROOT"]
        )
        + [
            third_party / "ThunderKittens",
            third_party / "thunderkittens",
            artifact_sources / "ThunderKittens",
        ],
    }
    return {name: _dedupe_paths(items) for name, items in roots.items()}


def _find_file(root: Path, names: Iterable[str], *, max_depth: int = 5) -> Optional[Path]:
    if not root.exists():
        return None
    wanted = set(names)
    for path in root.rglob("*"):
        try:
            rel_depth = len(path.relative_to(root).parts)
        except Exception:
            rel_depth = 999
        if rel_depth > max_depth:
            continue
        if path.is_file() and path.name in wanted:
            return path
    return None


def _inspect_cutlass(root: Path) -> dict[str, Any]:
    cute_header = _find_file(root, ["tensor.hpp"], max_depth=5)
    include_root = None
    if cute_header is not None:
        # Expected: <root>/include/cute/tensor.hpp.
        parts = cute_header.parts
        if "include" in parts:
            include_root = Path(*parts[: parts.index("include") + 1])
        elif cute_header.parent.name == "cute":
            include_root = cute_header.parent.parent
    return {
        "root": str(root),
        "exists": root.exists(),
        "cute_tensor_header": str(cute_header) if cute_header else None,
        "include_root": str(include_root) if include_root else None,
        "compile_candidate": include_root is not None,
    }


def _inspect_thunderkittens(root: Path) -> dict[str, Any]:
    header = _find_file(root, ["kittens.cuh"], max_depth=6)
    include_root = header.parent if header is not None else None
    return {
        "root": str(root),
        "exists": root.exists(),
        "kittens_header": str(header) if header else None,
        "include_root": str(include_root) if include_root else None,
        "compile_candidate": include_root is not None,
    }


def _nvcc_arch(args: argparse.Namespace) -> str:
    return args.cuda_arch or os.environ.get("STREAMATTN_CUDA_ARCH") or "sm_90a"


def _compile_cuda_translation_unit(
    *,
    name: str,
    source: str,
    include_dirs: list[Path],
    args: argparse.Namespace,
    cxx_standard: str,
    extra_flags: Optional[list[str]] = None,
) -> dict[str, Any]:
    nvcc = _which("nvcc")
    if not nvcc:
        return {"attempted": False, "available": False, "reason": "nvcc not found"}

    with tempfile.TemporaryDirectory(prefix=f"streamattn_{name}_") as tmp_text:
        tmp = Path(tmp_text)
        src = tmp / f"{name}.cu"
        obj = tmp / f"{name}.o"
        src.write_text(source, encoding="utf-8")
        cmd = [
            nvcc,
            f"-std={cxx_standard}",
            f"-arch={_nvcc_arch(args)}",
            "-O2",
            "-c",
            str(src),
            "-o",
            str(obj),
        ]
        if extra_flags:
            cmd.extend(extra_flags)
        for include_dir in include_dirs:
            cmd.extend(["-I", str(include_dir)])
        result = _run(cmd, timeout=args.compile_timeout_s)
        return {
            "attempted": True,
            "available": result["returncode"] == 0,
            "cmd": result["cmd"],
            "returncode": result["returncode"],
            "output_tail": result["output_tail"],
        }


def _compile_cutlass_smoke(info: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    include_root = info.get("include_root")
    if not include_root:
        return {"attempted": False, "available": False, "reason": "missing CuTe include root"}
    source = r"""
#include <cute/tensor.hpp>
#include <cute/layout.hpp>
#include <cute/numeric/integral_constant.hpp>

using namespace cute;

__global__ void streamattn_cute_smoke(float* out) {
  auto layout = make_layout(
      make_shape(Int<8>{}, Int<4>{}),
      make_stride(Int<4>{}, Int<1>{}));
  if (threadIdx.x == 0) {
    out[0] = float(size(layout));
  }
}
"""
    return _compile_cuda_translation_unit(
        name="cute_header_smoke",
        source=source,
        include_dirs=[Path(include_root)],
        args=args,
        cxx_standard="c++17",
    )


def _compile_thunderkittens_smoke(info: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    include_root = info.get("include_root")
    if not include_root:
        return {"attempted": False, "available": False, "reason": "missing kittens.cuh include root"}
    source = r"""
#include "kittens.cuh"

using namespace kittens;

__global__ void streamattn_tk_smoke(float* out) {
  if (threadIdx.x == 0) {
    out[0] = 1.0f;
  }
}
"""
    arch = _nvcc_arch(args)
    if arch in {"sm_90", "sm_90a", "compute_90", "compute_90a"}:
        tk_arch_define = "-DKITTENS_SM90"
    elif arch in {"sm_100", "sm_100a", "compute_100", "compute_100a"}:
        tk_arch_define = "-DKITTENS_SM100"
    elif arch in {"sm_103", "sm_103a", "compute_103", "compute_103a"}:
        tk_arch_define = "-DKITTENS_SM103"
    elif arch in {"sm_120", "sm_120a", "compute_120", "compute_120a"}:
        tk_arch_define = "-DKITTENS_SM120"
    else:
        tk_arch_define = "-DKITTENS_SM80"
    return _compile_cuda_translation_unit(
        name="thunderkittens_header_smoke",
        source=source,
        include_dirs=[Path(include_root)],
        args=args,
        cxx_standard="c++20",
        extra_flags=[
            tk_arch_define,
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
    )


def _first_compile_candidate(rows: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    for row in rows:
        if row.get("compile_candidate"):
            return row
    return None


def _decision(result: dict[str, Any]) -> dict[str, Any]:
    tk_compile = result.get("compile_smokes", {}).get("thunderkittens", {})
    cute_compile = result.get("compile_smokes", {}).get("cutlass_cute", {})
    flashinfer = result.get("flashinfer_context", {})

    tk_ok = bool(tk_compile.get("available"))
    cute_ok = bool(cute_compile.get("available"))
    flashinfer_sources = bool(flashinfer.get("installed_wheel_has_cuda_sources"))

    if tk_ok:
        path = "thunderkittens_head_mode_decode_spike"
        reason = "ThunderKittens headers compile with the current CUDA toolchain; use it for a tensor-core scheduler spike."
    elif cute_ok:
        path = "cute_head_mode_decode_spike"
        reason = "CuTe/CUTLASS headers compile; use CuTe for the custom tensor-core scheduler spike."
    elif flashinfer_sources:
        path = "flashinfer_scheduler_fork_spike"
        reason = "Editable FlashInfer CUDA sources appear available; patch the scheduler before K/V tile work."
    else:
        path = "fetch_tensor_core_backend_sources_then_retry"
        reason = "No tensor-core backend header smoke passed yet; fetch ThunderKittens or CUTLASS and rerun this probe."

    return {
        "recommended_next_path": path,
        "reason": reason,
        "do_not_spend_more_time_on": [
            "standalone_projection_scan",
            "flashinfer_custom_jit_math_hook",
            "python_stream_composition",
            "cuda_graph_around_current_components",
            "scalar_cuda_exact_branch",
            "scalar_cuda_splitk_exact_branch",
        ],
        "streamattn_unique_requirement": (
            "Per-Q-head SEED_ONLY must change the KV tiles scheduled before K load/QK; "
            "math-level masking after dense scheduling is too late."
        ),
    }


def probe(args: argparse.Namespace) -> dict[str, Any]:
    clones: dict[str, Any] = {}
    checkout_dir = Path(args.checkout_dir).expanduser()
    if args.clone_cutlass:
        clones["cutlass"] = _clone_repo(CUTLASS_REPO, checkout_dir / "cutlass")
    if args.clone_thunderkittens:
        clones["thunderkittens"] = _clone_repo(THUNDERKITTENS_REPO, checkout_dir / "ThunderKittens")

    roots = _candidate_roots(args)
    cutlass_rows = [_inspect_cutlass(root) for root in roots["cutlass"]]
    tk_rows = [_inspect_thunderkittens(root) for root in roots["thunderkittens"]]

    cutlass_candidate = _first_compile_candidate(cutlass_rows)
    tk_candidate = _first_compile_candidate(tk_rows)

    compile_smokes = {
        "cutlass_cute": (
            _compile_cutlass_smoke(cutlass_candidate, args)
            if args.compile_smoke and cutlass_candidate is not None
            else {"attempted": False, "available": None, "reason": "not requested or no candidate"}
        ),
        "thunderkittens": (
            _compile_thunderkittens_smoke(tk_candidate, args)
            if args.compile_smoke and tk_candidate is not None
            else {"attempted": False, "available": None, "reason": "not requested or no candidate"}
        ),
    }

    flashinfer_context = {}
    try:
        import flashinfer  # type: ignore

        root = Path(flashinfer.__file__).resolve().parent
        source_like = [
            str(path.relative_to(root)).replace("\\", "/")
            for path in root.rglob("*")
            if path.is_file() and path.suffix in {".cu", ".cuh", ".h", ".hpp", ".cc", ".cpp"}
        ]
        flashinfer_context = {
            "available": True,
            "version": getattr(flashinfer, "__version__", None),
            "root": str(root),
            "source_like_files_sample": source_like[:80],
            "installed_wheel_has_cuda_sources": any(
                item.endswith((".cu", ".cuh")) for item in source_like
            ),
        }
    except Exception as exc:
        flashinfer_context = {"available": False, "error": repr(exc)}

    result: dict[str, Any] = {
        "schema": "streamattn.tensor_core_backend_paths.v1",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "nvcc": _which("nvcc"),
            "git": _which("git"),
            "cuda_arch": _nvcc_arch(args),
            "nvcc_version": _version_cmd(["nvcc", "--version"]),
            "gxx_version": _version_cmd(["g++", "--version"]),
        },
        "clone_results": clones,
        "candidates": {
            "cutlass": cutlass_rows,
            "thunderkittens": tk_rows,
        },
        "compile_smokes": compile_smokes,
        "flashinfer_context": flashinfer_context,
    }
    result["decision"] = _decision(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout-dir", default="artifacts/backend_sources")
    parser.add_argument("--clone-cutlass", action="store_true")
    parser.add_argument("--clone-thunderkittens", action="store_true")
    parser.add_argument("--compile-smoke", action="store_true")
    parser.add_argument("--compile-timeout-s", type=int, default=900)
    parser.add_argument("--cuda-arch", default="")
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
