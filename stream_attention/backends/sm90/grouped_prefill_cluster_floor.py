"""Planning-only SM90 two-CTA TMA multicast transport floor."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Optional

import torch

from .grouped_prefill_cluster_floor_sources import CPP_SOURCE, CUDA_SOURCE
from .transposed_gqa_exact import resolve_cutlass_root


_EXTENSIONS: dict[tuple[str, str], Any] = {}
_EXTENSION_LOCK = threading.Lock()
RESOURCE_FIELDS = (
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "local_bytes_per_thread",
    "blocks_per_sm",
    "max_threads_per_block",
)


def compile_grouped_prefill_cluster_floor_extension(
    *,
    cutlass_root: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    verbose: bool = False,
):
    """Compile and cache the isolated SM90 cluster transport floor."""

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
                name=f"streamattn_sm90_prefill_cluster_floor_{source_id}",
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


def decode_cluster_resource_info(
    values: torch.Tensor,
) -> dict[str, dict[str, int]]:
    """Decode paired independent and multicast resource telemetry."""

    raw = [int(value) for value in values.cpu().tolist()]
    width = len(RESOURCE_FIELDS)
    if len(raw) != 2 * width:
        raise ValueError(f"expected {2 * width} resource values, got {len(raw)}")
    return {
        "independent": dict(zip(RESOURCE_FIELDS, raw[:width])),
        "multicast": dict(zip(RESOURCE_FIELDS, raw[width:])),
    }
