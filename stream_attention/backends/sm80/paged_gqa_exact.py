"""Native Ampere paged GQA decode extension."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Optional

from .paged_gqa_exact_sources import CPP_SOURCE, CUDA_SOURCE


_EXTENSIONS: dict[tuple[str, str], Any] = {}
_EXTENSION_LOCK = threading.Lock()


def sm80_paged_gqa_source_id() -> str:
    """Return the immutable CUDA/C++ source identity used in evidence keys."""

    return hashlib.sha1((CPP_SOURCE + CUDA_SOURCE).encode("utf-8")).hexdigest()[:12]


def _cutlass_candidates(explicit: Optional[Path] = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    for name in ("STREAMATTN_CUTLASS_ROOT", "CUTLASS_ROOT", "CUTLASS_PATH"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value))
    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            repo_root / "artifacts/backend_sources/FlashMLA-ETAP/csrc/cutlass",
            Path("/opt/flashmla-etap/csrc/cutlass"),
        ]
    )
    return candidates


def resolve_cutlass_root(explicit: Optional[Path] = None) -> Path:
    for candidate in _cutlass_candidates(explicit):
        resolved = candidate.expanduser().resolve()
        if (resolved / "include/cute/tensor.hpp").is_file():
            return resolved
    raise FileNotFoundError(
        "CUTLASS headers were not found; set STREAMATTN_CUTLASS_ROOT to a "
        "CUTLASS tree containing include/cute/tensor.hpp"
    )


def compile_sm80_paged_gqa_extension(
    *,
    cutlass_root: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    verbose: bool = False,
):
    """Compile and cache the SM80 page-16 HND/NHD D128/G8 extension."""

    from torch.utils.cpp_extension import load_inline

    resolved_cutlass = resolve_cutlass_root(cutlass_root)
    if build_dir is None and os.environ.get("STREAMATTN_SM80_BUILD_DIR"):
        build_dir = Path(os.environ["STREAMATTN_SM80_BUILD_DIR"])
    resolved_build = (
        str(Path(build_dir).expanduser().resolve()) if build_dir is not None else ""
    )
    key = (str(resolved_cutlass), resolved_build)
    with _EXTENSION_LOCK:
        cached = _EXTENSIONS.get(key)
        if cached is not None:
            return cached

        source_id = hashlib.sha1(
            (sm80_paged_gqa_source_id() + key[0]).encode("utf-8")
        ).hexdigest()[:12]
        kwargs: dict[str, Any] = {}
        if build_dir is not None:
            resolved_build_path = Path(resolved_build)
            resolved_build_path.mkdir(parents=True, exist_ok=True)
            kwargs["build_directory"] = str(resolved_build_path)

        previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
        try:
            extension = load_inline(
                name=f"streamattn_sm80_paged_gqa_{source_id}",
                cpp_sources=CPP_SOURCE,
                cuda_sources=CUDA_SOURCE,
                extra_include_paths=[str(resolved_cutlass / "include")],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=[
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-gencode=arch=compute_80,code=sm_80",
                ],
                with_cuda=True,
                verbose=verbose,
                **kwargs,
            )
        finally:
            if previous_arch is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
        _EXTENSIONS[key] = extension
        return extension
