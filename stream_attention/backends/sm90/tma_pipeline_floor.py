"""Planning-only SM90 D128 TMA pipeline floor extension."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Optional

import torch

from .tma_pipeline_floor_sources import CPP_SOURCE, CUDA_SOURCE
from .transposed_gqa_exact import resolve_cutlass_root


_EXTENSIONS: dict[tuple[str, str], Any] = {}
_EXTENSION_LOCK = threading.Lock()


RESOURCE_FIELDS = (
    "tile_bytes",
    "cp_storage_bytes",
    "tma_2k_storage_bytes",
    "tma_2k1v_storage_bytes",
    "raw_2k_bytes",
    "raw_2k1v_bytes",
    "raw_2k2v_bytes",
)
KERNEL_RESOURCE_FIELDS = (
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "blocks_per_sm",
    "max_threads_per_block",
)
KERNEL_NAMES = ("cp_async_k", "cp_async_kv", "tma_k", "tma_kv_2k1v")


def compile_tma_pipeline_floor_extension(
    *,
    cutlass_root: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    verbose: bool = False,
):
    """Compile and cache the isolated SM90 TMA floor extension."""

    from torch.utils.cpp_extension import load_inline

    resolved_cutlass = resolve_cutlass_root(cutlass_root)
    resolved_build = (
        str(Path(build_dir).expanduser().resolve()) if build_dir is not None else ""
    )
    key = (str(resolved_cutlass), resolved_build)
    with _EXTENSION_LOCK:
        cached = _EXTENSIONS.get(key)
        if cached is not None:
            return cached

        source_id = hashlib.sha1(
            (CPP_SOURCE + CUDA_SOURCE + key[0]).encode("utf-8")
        ).hexdigest()[:12]
        kwargs: dict[str, Any] = {}
        if build_dir is not None:
            path = Path(resolved_build)
            path.mkdir(parents=True, exist_ok=True)
            kwargs["build_directory"] = str(path)

        previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        try:
            extension = load_inline(
                name=f"streamattn_sm90_tma_floor_{source_id}",
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
                    "--ptxas-options=-v",
                    "-gencode=arch=compute_90a,code=sm_90a",
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


def decode_resource_info(values: torch.Tensor) -> dict[str, Any]:
    """Decode the compact C++ resource vector into named fields."""

    raw = [int(value) for value in values.cpu().tolist()]
    expected = len(RESOURCE_FIELDS) + len(KERNEL_NAMES) * len(KERNEL_RESOURCE_FIELDS)
    if len(raw) != expected:
        raise ValueError(f"expected {expected} resource values, got {len(raw)}")
    result: dict[str, Any] = dict(zip(RESOURCE_FIELDS, raw[: len(RESOURCE_FIELDS)]))
    offset = len(RESOURCE_FIELDS)
    result["kernels"] = {}
    for name in KERNEL_NAMES:
        chunk = raw[offset : offset + len(KERNEL_RESOURCE_FIELDS)]
        result["kernels"][name] = dict(zip(KERNEL_RESOURCE_FIELDS, chunk))
        offset += len(KERNEL_RESOURCE_FIELDS)
    return result
