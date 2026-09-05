"""Compile-time dtype/mask selection; mutable logical positions stay on device."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

import torch

from .micro_prefill_semantics_sources import CPP_SOURCE, semantic_cuda_source
from .transposed_gqa_exact import resolve_cutlass_root


_EXTENSIONS: dict[tuple[str, str], Any] = {}
_LOCK = threading.Lock()


def validate_positions(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    causal: bool,
    query_positions: torch.Tensor | None,
    key_positions: torch.Tensor | None,
) -> None:
    """Do not infer top-left or bottom-right alignment from tensor lengths.

    For causal execution key_positions[b,j] <= query_positions[b,i] is the
    complete visibility rule. Positions need not be sorted or start at zero.
    Shapes, dtype and device are checked without reading any device values.
    """
    if not isinstance(causal, bool):
        raise ValueError("causal must be a bool")
    if not causal:
        if query_positions is not None or key_positions is not None:
            raise ValueError("positions require causal=True")
        return
    for name, tensor, shape in (
        ("query_positions", query_positions, query.shape[:2]),
        ("key_positions", key_positions, (key.shape[0], key.shape[2])),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"causal=True requires explicit {name}")
        if tuple(tensor.shape) != tuple(shape) or tensor.dtype != torch.int64:
            raise ValueError(f"{name} must be int64 with shape {tuple(shape)}")
        if tensor.device != query.device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous on the query device")


def compile_semantic_extension(
    *, head_dim: int, dtype: torch.dtype, causal: bool,
    cutlass_root: Path | None = None, build_dir: Path | None = None,
    verbose: bool = False,
) -> Any:
    from torch.utils.cpp_extension import get_default_build_root, load_inline

    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("micro-prefill supports BF16 and FP16")
    source = semantic_cuda_source(
        head_dim, "bf16" if dtype == torch.bfloat16 else "fp16", causal
    )
    root = resolve_cutlass_root(cutlass_root)
    flags = [
        "-O3", "-std=c++17", "--use_fast_math", "--expt-relaxed-constexpr",
        "--expt-extended-lambda", "--ptxas-options=-v,--warn-on-spills",
        "-lineinfo", "--keep", "--keep-dir=.",
        "-gencode=arch=compute_90a,code=sm_90a",
    ]
    identity = hashlib.sha256(
        (CPP_SOURCE + source + str(root) + repr(flags)).encode()
    ).hexdigest()[:16]
    name = "streamattn_sm90_micro_semantics_" + identity
    directory = (Path(build_dir) if build_dir else Path(get_default_build_root())) / name
    directory = directory.resolve()
    cache_key = (str(directory), identity)
    with _LOCK:
        if cache_key in _EXTENSIONS:
            return _EXTENSIONS[cache_key]
        directory.mkdir(parents=True, exist_ok=True)
        previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        try:
            extension = load_inline(
                name=name, cpp_sources=CPP_SOURCE, cuda_sources=source,
                build_directory=str(directory),
                extra_include_paths=[str(root / "include")],
                extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=flags,
                with_cuda=True, verbose=verbose,
            )
        finally:
            if previous is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous
        _EXTENSIONS[cache_key] = extension
        return extension
