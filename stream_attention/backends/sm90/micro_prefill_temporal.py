"""Allocation-free, isolated R64 temporal canary; never a public dispatch route.

``temporal`` overlaps next-score softmax with current PV. ``drained`` changes
only that partial wait to a full drain. Both retain the original partial ABI
and merge, including a real merge at S1. No public LSE buffer is allocated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
from typing import Any

import torch

from .micro_prefill import micro_prefill_shape_reasons
from .micro_prefill_temporal_sources import CPP_SOURCE, cuda_source_for_head_dim
from .transposed_gqa_exact import resolve_cutlass_root


QUERY_TILE_ROWS = 64
KV_TILE_TOKENS = 64
PROTOCOLS = {"temporal": 0, "drained": 1}
COMPONENTS = {"combined": 0, "producer": 1, "merge": 2}
RESOURCE_FIELDS = (
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "local_bytes_per_thread",
    "blocks_per_sm",
    "max_threads_per_block",
)
_EXTENSIONS: dict[tuple[str, str], Any] = {}
_EXTENSION_LOCK = threading.Lock()


def query_tiles_temporal(query_len: int, group_size: int) -> int:
    if not 1 <= query_len <= 64 or group_size not in (4, 8):
        raise ValueError("require M in [1,64] and G in {4,8}")
    return (query_len * group_size + QUERY_TILE_ROWS - 1) // QUERY_TILE_ROWS


def balanced_tile_interval(tiles: int, splits: int, split: int) -> tuple[int, int]:
    if not 1 <= splits <= min(tiles, 512) or not 0 <= split < splits:
        raise ValueError("require 1 <= S <= min(T,512) and 0 <= split < S")
    return split * tiles // splits, (split + 1) * tiles // splits


def supports_sm90_temporal(
    query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor
) -> bool:
    return (
        not temporal_shape_reasons(query, key_cache, value_cache)
        and query.is_cuda
        and torch.cuda.get_device_capability(query.device) == (9, 0)
    )


def temporal_shape_reasons(
    query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor
) -> list[str]:
    reasons = micro_prefill_shape_reasons(query, key_cache, value_cache)
    # The same packed-row geometry also represents a single query position.
    if query.dim() == 4 and query.shape[1] == 1:
        reasons = [reason for reason in reasons if reason != "query_len"]
    return reasons


def source_fingerprint(head_dim: int) -> str:
    return hashlib.sha256(
        (CPP_SOURCE + cuda_source_for_head_dim(head_dim)).encode("utf-8")
    ).hexdigest()


def cuda_compile_flags(*, diagnostic_build: bool = False) -> list[str]:
    flags = [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--ptxas-options=-v,--warn-on-spills",
        "-gencode=arch=compute_90a,code=sm_90a",
    ]
    if diagnostic_build:
        # Ninja compiles in this extension's isolated directory, even with spaces.
        flags += ["-lineinfo", "--keep", "--keep-dir=."]
    return flags


def compile_micro_prefill_temporal_extension(
    *,
    head_dim: int,
    cutlass_root: Path | None = None,
    build_dir: Path | None = None,
    verbose: bool = False,
    diagnostic_build: bool = False,
) -> Any:
    """JIT only during planning; diagnostic PTX is beside extension.__file__."""
    from torch.utils.cpp_extension import get_default_build_root, load_inline

    source = cuda_source_for_head_dim(head_dim)
    root = resolve_cutlass_root(cutlass_root)
    flags = cuda_compile_flags(diagnostic_build=diagnostic_build)
    identity = hashlib.sha256(
        (source_fingerprint(head_dim) + str(root) + repr(flags)).encode("utf-8")
    ).hexdigest()[:16]
    name = "streamattn_sm90_temporal_" + identity
    base = Path(build_dir) if build_dir is not None else Path(get_default_build_root())
    directory = base.resolve() / name
    cache_key = (str(directory), identity)
    with _EXTENSION_LOCK:
        if cache_key in _EXTENSIONS:
            return _EXTENSIONS[cache_key]
        directory.mkdir(parents=True, exist_ok=True)
        previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        try:
            extension = load_inline(
                name=name,
                cpp_sources=CPP_SOURCE,
                cuda_sources=source,
                build_directory=str(directory),
                extra_include_paths=[str(root / "include")],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=flags,
                with_cuda=True,
                verbose=verbose,
            )
        finally:
            if previous_arch is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
        _EXTENSIONS[cache_key] = extension
        return extension


def decode_resource_info(values: torch.Tensor) -> dict[str, dict[str, int]]:
    raw = [int(value) for value in values.cpu().tolist()]
    width = len(RESOURCE_FIELDS)
    if len(raw) != 2 * width:
        raise ValueError(f"expected {2 * width} resource values, got {len(raw)}")
    return {
        name: dict(zip(RESOURCE_FIELDS, raw[index * width : (index + 1) * width]))
        for index, name in enumerate(("producer", "merge"))
    }


def resource_gate(resources: dict[str, dict[str, int]]) -> dict[str, bool]:
    zero_local = all(
        resources[name]["local_bytes_per_thread"] == 0 for name in ("producer", "merge")
    )
    producer_resident = resources["producer"]["blocks_per_sm"] >= 2
    merge_resident = resources["merge"]["blocks_per_sm"] >= 1
    return dict(
        zero_local_bytes=zero_local,
        two_resident_ctas=producer_resident,
        merge_resident=merge_resident,
        passed=zero_local and producer_resident and merge_resident,
    )


@dataclass
class TemporalMicroPrefillPlan:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    output: torch.Tensor
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    num_splits: int
    query_tiles: int
    protocol: str
    extension: Any
    resources: dict[str, dict[str, int]]
    backend: str = "sm90_r64_temporal_rs_canary"

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
        num_splits: int | None = None,
        protocol: str = "temporal",
        target_producer_ctas: int = 256,
        cutlass_root: Path | None = None,
        build_dir: Path | None = None,
        compile_verbose: bool = False,
        diagnostic_build: bool = False,
    ) -> "TemporalMicroPrefillPlan":
        if protocol not in PROTOCOLS:
            raise ValueError("protocol must be temporal or drained")
        reasons = temporal_shape_reasons(query, key_cache, value_cache)
        if reasons:
            raise ValueError("unsupported temporal buffers: " + ",".join(reasons))
        if not supports_sm90_temporal(query, key_cache, value_cache):
            raise ValueError("temporal canary requires SM90 CUDA buffers")
        b, m, hq, dim = map(int, query.shape)
        hk, n = int(key_cache.shape[1]), int(key_cache.shape[2])
        tiles = query_tiles_temporal(m, hq // hk)
        groups = b * hk * tiles
        if (
            not isinstance(target_producer_ctas, int)
            or isinstance(target_producer_ctas, bool)
            or target_producer_ctas <= 0
        ):
            raise ValueError("target_producer_ctas must be a positive integer")
        splits = num_splits
        if splits is None:
            splits = min(512, n // 64, (target_producer_ctas + groups - 1) // groups)
        if (
            not isinstance(splits, int)
            or isinstance(splits, bool)
            or not 1 <= splits <= min(n // 64, 512)
        ):
            raise ValueError("num_splits must be an integer in [1,min(N/64,512)]")
        if max(groups * splits, b * m * hq, n) > 2**31 - 1:
            raise ValueError("grid or KV extent exceeds int32")
        if output is None:
            output = torch.empty_like(query)
        if output.shape != query.shape or output.dtype != query.dtype:
            raise ValueError("output must match query shape and dtype")
        if output.device != query.device or not output.is_contiguous():
            raise ValueError("output must be contiguous on the query device")
        partial_output = torch.empty(
            (groups, splits, QUERY_TILE_ROWS, dim),
            dtype=torch.float32,
            device=query.device,
        )
        partial_lse = torch.empty(
            (groups, splits, QUERY_TILE_ROWS), dtype=torch.float32, device=query.device
        )
        extension = compile_micro_prefill_temporal_extension(
            head_dim=dim,
            cutlass_root=cutlass_root,
            build_dir=build_dir,
            verbose=compile_verbose,
            diagnostic_build=diagnostic_build,
        )
        # Opt into dynamic shared memory and query resources before capture.
        resources = decode_resource_info(
            extension.resource_info(query, PROTOCOLS[protocol])
        )
        return cls(
            query,
            key_cache,
            value_cache,
            output,
            partial_output,
            partial_lse,
            splits,
            tiles,
            protocol,
            extension,
            resources,
        )

    @property
    def workspace_bytes(self) -> int:
        return 4 * (self.partial_output.numel() + self.partial_lse.numel())

    @property
    def producer_ctas(self) -> int:
        return (
            int(self.query.shape[0] * self.key_cache.shape[1])
            * self.query_tiles
            * self.num_splits
        )

    @property
    def resource_pass(self) -> bool:
        return resource_gate(self.resources)["passed"]

    def run_component(self, component: str | int = "combined") -> torch.Tensor:
        which = (
            COMPONENTS.get(component, -1) if isinstance(component, str) else component
        )
        if (
            not isinstance(which, int)
            or isinstance(which, bool)
            or which not in (0, 1, 2)
        ):
            raise ValueError("component must be combined, producer, or merge")
        self.extension.out(
            self.query,
            self.key_cache,
            self.value_cache,
            self.partial_output,
            self.partial_lse,
            self.output,
            self.num_splits,
            which,
            PROTOCOLS[self.protocol],
        )
        return self.output

    def run(self) -> torch.Tensor:
        """Fixed buffers and source: no JIT, resource queries, or tensor allocations."""
        return self.run_component("combined")


__all__ = [
    "TemporalMicroPrefillPlan",
    "PROTOCOLS",
    "QUERY_TILE_ROWS",
    "KV_TILE_TOKENS",
    "query_tiles_temporal",
    "balanced_tile_interval",
    "source_fingerprint",
    "cuda_compile_flags",
    "compile_micro_prefill_temporal_extension",
    "decode_resource_info",
    "resource_gate",
    "temporal_shape_reasons",
    "supports_sm90_temporal",
]
