"""Diagnostic SM90 grouped-prefill attention-epoch floors.

The extension compares shared-sourced and register-sourced PV without touching
the promoted dispatch surface.  It is deliberately a component experiment:
TMA and multi-warpgroup overlap are separate promotion gates.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Optional

import torch

from .grouped_prefill_epoch_floor_sources import CPP_SOURCE, CUDA_SOURCE
from .transposed_gqa_exact import resolve_cutlass_root


_EXTENSIONS: dict[tuple[str, str], Any] = {}
_EXTENSION_LOCK = threading.Lock()

KERNEL_NAMES = (
    "qk",
    "qk_softmax",
    "pv_ss",
    "pv_rs",
    "epoch_ss",
    "epoch_rs",
    "epoch_rs_reuse_q",
)
KERNEL_RESOURCE_FIELDS = (
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "local_bytes_per_thread",
    "blocks_per_sm",
    "max_threads_per_block",
)
RESOURCE_FIELDS = (
    "qk_shared_bytes",
    "pv_ss_shared_bytes",
    "pv_rs_shared_bytes",
    "epoch_ss_shared_bytes",
    "epoch_rs_shared_bytes",
    "tma_epoch_shared_bytes",
)


def compile_grouped_prefill_epoch_floor_extension(
    *,
    cutlass_root: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    verbose: bool = False,
):
    """Compile and cache the isolated SM90 attention-epoch floor extension."""

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
                name=f"streamattn_sm90_prefill_epoch_floor_{source_id}",
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
    """Decode the extension's compact resource vector."""

    raw = [int(value) for value in values.cpu().tolist()]
    expected = len(RESOURCE_FIELDS) + len(KERNEL_NAMES) * len(
        KERNEL_RESOURCE_FIELDS
    )
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


def decode_tma_resource_info(values: torch.Tensor) -> dict[str, int]:
    """Decode resource telemetry for the descriptor-specialized TMA kernel."""

    raw = [int(value) for value in values.cpu().tolist()]
    if len(raw) != len(KERNEL_RESOURCE_FIELDS):
        raise ValueError(
            f"expected {len(KERNEL_RESOURCE_FIELDS)} TMA resource values, got {len(raw)}"
        )
    return dict(zip(KERNEL_RESOURCE_FIELDS, raw))


def decode_grouped2_resource_info(values: torch.Tensor) -> dict[str, dict[str, int]]:
    """Decode telemetry for the paired serial and grouped TMA kernels."""

    raw = [int(value) for value in values.cpu().tolist()]
    expected = 2 * len(KERNEL_RESOURCE_FIELDS)
    if len(raw) != expected:
        raise ValueError(f"expected {expected} grouped resource values, got {len(raw)}")
    width = len(KERNEL_RESOURCE_FIELDS)
    return {
        "serial_grouped2": dict(zip(KERNEL_RESOURCE_FIELDS, raw[:width])),
        "tma_grouped2": dict(zip(KERNEL_RESOURCE_FIELDS, raw[width:])),
    }


def decode_cluster2_epoch_resource_info(
    values: torch.Tensor,
) -> dict[str, dict[str, int]]:
    """Decode independent and multicast cluster attention-epoch telemetry."""

    raw = [int(value) for value in values.cpu().tolist()]
    expected = 2 * len(KERNEL_RESOURCE_FIELDS)
    if len(raw) != expected:
        raise ValueError(
            f"expected {expected} cluster epoch resource values, got {len(raw)}"
        )
    width = len(KERNEL_RESOURCE_FIELDS)
    return {
        "independent": dict(zip(KERNEL_RESOURCE_FIELDS, raw[:width])),
        "multicast": dict(zip(KERNEL_RESOURCE_FIELDS, raw[width:])),
    }
