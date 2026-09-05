"""Experimental, allocation-free M128 attention plans, never public dispatch.

Rows pack query positions and GQA heads. The default split count matches the
existing M64 family, rather than doubling splits to preserve its CTA count.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
from typing import Any

import torch

from .micro_prefill import (
    choose_natural_micro_prefill_splits,
    micro_prefill_shape_reasons,
    supports_sm90_micro_prefill,
)
from .micro_prefill_128_sources import CPP_SOURCE, cuda_source_for_head_dim
from .transposed_gqa_exact import resolve_cutlass_root


QUERY_TILE_ROWS = 128
KV_TILE_TOKENS = 64
PROTOCOLS = {"overlap": 0, "serial": 1, "overlap_drained": 2}
COMPONENTS = {"combined": 0, "producer": 1, "merge": 2}
RESOURCE_FIELDS = (
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "local_bytes_per_thread",
    "blocks_per_sm",
    "max_threads_per_block",
)
_EXTENSIONS: dict[tuple[str, str, str, bool, bool], Any] = {}
_EXTENSION_LOCK = threading.Lock()


def query_tiles_128(query_len: int, group_size: int) -> int:
    if not 2 <= query_len <= 64:
        raise ValueError("query_len must be in [2,64]")
    if group_size not in (4, 8):
        raise ValueError("group_size must be 4 or 8")
    return (query_len * group_size + QUERY_TILE_ROWS - 1) // QUERY_TILE_ROWS


def balanced_tile_interval(tiles: int, splits: int, split: int) -> tuple[int, int]:
    if not 1 <= splits <= tiles or not 0 <= split < splits:
        raise ValueError("require 1 <= splits <= tiles and 0 <= split < splits")
    return split * tiles // splits, (split + 1) * tiles // splits


def choose_micro_prefill_128_splits(
    *,
    batch: int,
    query_len: int,
    kv_heads: int,
    group_size: int,
    kv_len: int,
    target_producer_ctas: int = 256,
) -> int:
    """Match M64 splits; this is a controlled default, not a tuned selector."""
    return min(
        512,
        choose_natural_micro_prefill_splits(
            batch=batch,
            query_len=query_len,
            kv_heads=kv_heads,
            group_size=group_size,
            kv_len=kv_len,
            target_producer_ctas=target_producer_ctas,
        ),
    )


def source_fingerprint(head_dim: int) -> str:
    return hashlib.sha256(
        (CPP_SOURCE + cuda_source_for_head_dim(head_dim)).encode("utf-8")
    ).hexdigest()


def compile_micro_prefill_128_extension(
    *,
    head_dim: int,
    cutlass_root: Path | None = None,
    build_dir: Path | None = None,
    verbose: bool = False,
    lineinfo: bool = False,
    keep_intermediates: bool = False,
    diagnostic_build: bool = False,
) -> Any:
    """Compile during planning; diagnostic flags never share a normal build cache."""
    from torch.utils.cpp_extension import CUDA_HOME, _get_build_directory, load_inline

    keep_intermediates = keep_intermediates or diagnostic_build
    source = cuda_source_for_head_dim(head_dim)
    root = resolve_cutlass_root(cutlass_root)
    source_id = source_fingerprint(head_dim)
    extension_name = (
        "streamattn_sm90_micro128_"
        + hashlib.sha256(
            f"{source_id}|{root}|lineinfo={lineinfo}|keep={keep_intermediates}".encode()
        ).hexdigest()[:16]
    )
    # Distinct directories allow M64/M128 and D64/D128 builds to coexist.
    directory = (
        Path(build_dir).resolve() / extension_name if build_dir is not None else None
    )
    key = (str(root), str(directory or ""), source_id, lineinfo, keep_intermediates)
    with _EXTENSION_LOCK:
        if key in _EXTENSIONS:
            return _EXTENSIONS[key]
        kwargs: dict[str, Any] = {}
        if keep_intermediates and directory is None:
            directory = Path(_get_build_directory(extension_name, verbose)).resolve()
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            kwargs["build_directory"] = str(directory)
        cflags = ["-O3", "-std=c++17"]
        cuda_flags = [
            "-O3",
            "-std=c++17",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--ptxas-options=-v,--warn-on-spills",
            "-gencode=arch=compute_90a,code=sm_90a",
        ]
        if lineinfo:
            cuda_flags.append("-lineinfo")
        intermediates = directory / "nvcc_intermediates" if keep_intermediates else None
        if intermediates is not None:
            intermediates.mkdir(parents=True, exist_ok=True)
            cuda_flags += ["--keep", "--keep-dir", str(intermediates)]
        previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        try:
            extension = load_inline(
                name=extension_name,
                cpp_sources=CPP_SOURCE,
                cuda_sources=source,
                extra_include_paths=[str(root / "include")],
                extra_cflags=cflags,
                extra_cuda_cflags=cuda_flags,
                with_cuda=True,
                keep_intermediates=True,
                verbose=verbose,
                **kwargs,
            )
        finally:
            if previous_arch is None:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch
        binary_path = getattr(extension, "__file__", None)
        loaded_binary = None
        if binary_path is not None:
            binary = Path(binary_path).resolve()
            loaded_binary = {
                "path": str(binary),
                "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            }
        extension._streamattn_build_metadata = {
            "extension_name": extension_name,
            "source_sha256": source_id,
            "head_dim": head_dim,
            "cutlass_root": str(root),
            "cuda_home": CUDA_HOME,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "extra_cflags": cflags,
            "extra_cuda_cflags": cuda_flags,
            "lineinfo": lineinfo,
            "keep_intermediates": keep_intermediates,
            "diagnostic_build": keep_intermediates,
            "loaded_binary": loaded_binary,
            "intermediates_dir": str(intermediates) if intermediates else None,
            "build_directory": str(directory) if directory else None,
        }
        _EXTENSIONS[key] = extension
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


def resource_gate(
    resources: dict[str, dict[str, int]], *, direct: bool
) -> dict[str, bool]:
    active = [resources["producer"]] + ([] if direct else [resources["merge"]])
    zero_local = all(row["local_bytes_per_thread"] == 0 for row in active)
    resident = resources["producer"]["blocks_per_sm"] >= 2
    merge_resident = direct or resources["merge"]["blocks_per_sm"] >= 1
    return dict(
        zero_local_bytes=zero_local,
        two_resident_ctas=resident,
        merge_resident=merge_resident,
        passed=zero_local and resident and merge_resident,
    )


def _check_buffer(
    tensor: torch.Tensor,
    query: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> None:
    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise ValueError(f"{name} shape/dtype mismatch")
    if tensor.device != query.device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous on the query device")


@dataclass
class Natural128AsyncMicroPrefillPlan:
    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    num_splits: int
    query_tiles: int
    protocol: str
    direct: bool
    extension: Any
    resources: dict[str, dict[str, int]]
    backend: str = "sm90_natural_m128_intra_wg_rs_canary"

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        output: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
        num_splits: int | None = None,
        protocol: str = "overlap",
        direct: bool | None = None,
        cutlass_root: Path | None = None,
        build_dir: Path | None = None,
        compile_verbose: bool = False,
        lineinfo: bool = False,
        keep_intermediates: bool = False,
        diagnostic_build: bool = False,
    ) -> "Natural128AsyncMicroPrefillPlan":
        if protocol not in PROTOCOLS:
            raise ValueError("protocol must be overlap, serial, or overlap_drained")
        reasons = micro_prefill_shape_reasons(query, key_cache, value_cache)
        if query.dtype != torch.bfloat16 and "dtype" not in reasons:
            reasons.append("dtype")
        if reasons:
            raise ValueError("unsupported M128 buffers: " + ",".join(reasons))
        if not supports_sm90_micro_prefill(
            query, key_cache, value_cache, require_cutlass=False
        ):
            raise ValueError("M128 canary requires SM90 CUDA buffers")
        b, m, hq, dim = map(int, query.shape)
        hk, n = int(key_cache.shape[1]), int(key_cache.shape[2])
        group = hq // hk
        tiles = query_tiles_128(m, group)
        splits = (
            num_splits
            if num_splits is not None
            else choose_micro_prefill_128_splits(
                batch=b,
                query_len=m,
                kv_heads=hk,
                group_size=group,
                kv_len=n,
            )
        )
        if (
            not isinstance(splits, int)
            or isinstance(splits, bool)
            or not 1 <= splits <= min(n // 64, 512)
        ):
            raise ValueError("num_splits must be an integer in [1,min(N/64,512)]")
        use_direct = splits == 1 if direct is None else direct
        if not isinstance(use_direct, bool) or (use_direct and splits != 1):
            raise ValueError("direct mode requires num_splits=1")
        if max(b * hk * tiles * splits, b * m * hq, n) > 2**31 - 1:
            raise ValueError("grid or KV extent exceeds int32")
        if output is None:
            output = torch.empty_like(query)
        if lse is None:
            lse = torch.empty((b, m, hq), dtype=torch.float32, device=query.device)
        _check_buffer(output, query, (b, m, hq, dim), query.dtype, "output")
        _check_buffer(lse, query, (b, m, hq), torch.float32, "lse")
        groups = b * hk * tiles
        partial_output = torch.empty(
            (0,) if use_direct else (groups, splits, QUERY_TILE_ROWS, dim),
            dtype=torch.float32,
            device=query.device,
        )
        partial_lse = torch.empty(
            (0,) if use_direct else (groups, splits, QUERY_TILE_ROWS),
            dtype=torch.float32,
            device=query.device,
        )
        extension = compile_micro_prefill_128_extension(
            head_dim=dim,
            cutlass_root=cutlass_root,
            build_dir=build_dir,
            verbose=compile_verbose,
            lineinfo=lineinfo,
            keep_intermediates=keep_intermediates,
            diagnostic_build=diagnostic_build,
        )
        # Also opts into dynamic shared memory, outside graph capture/replay.
        resources = decode_resource_info(
            extension.resource_info(
                query,
                group,
                PROTOCOLS[protocol],
                use_direct,
            )
        )
        return cls(
            query,
            key_cache,
            value_cache,
            output,
            lse,
            partial_output,
            partial_lse,
            splits,
            tiles,
            protocol,
            use_direct,
            extension,
            resources,
        )

    @property
    def workspace_bytes(self) -> int:
        """Partial-state scratch only; output and public natural-log LSE are not scratch."""
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
        return resource_gate(self.resources, direct=self.direct)["passed"]

    @property
    def kernel_names(self) -> dict[str, str]:
        """Exact demangled identities for binary inspection, not profiler wildcards."""
        group = int(self.query.shape[2] // self.key_cache.shape[1])
        overlap = str(self.protocol != "serial").lower()
        direct = str(self.direct).lower()
        drained = str(self.protocol == "overlap_drained").lower()
        names = {
            "producer": (
                f"streamattn_micro128_kernel<{group}, {overlap}, {direct}, {drained}>"
            )
        }
        if not self.direct:
            names["merge"] = "streamattn_micro128_merge_kernel"
        return names

    def run_component(self, component: str | int = "combined") -> torch.Tensor:
        which = (
            COMPONENTS.get(component, -1) if isinstance(component, str) else component
        )
        if which not in (0, 1, 2):
            raise ValueError("component must be combined, producer, or merge")
        if self.direct and which == 2:
            raise ValueError("direct mode has no merge component")
        self.extension.out(
            self.query,
            self.key_cache,
            self.value_cache,
            self.partial_output,
            self.partial_lse,
            self.output,
            self.lse,
            self.num_splits,
            which,
            PROTOCOLS[self.protocol],
            self.direct,
        )
        return self.output

    def run(self) -> torch.Tensor:
        """Replay using fixed buffers; no compilation, resource query, or tensor allocation."""
        return self.run_component(0)


__all__ = [
    "Natural128AsyncMicroPrefillPlan",
    "query_tiles_128",
    "balanced_tile_interval",
    "choose_micro_prefill_128_splits",
    "compile_micro_prefill_128_extension",
    "decode_resource_info",
    "resource_gate",
    "source_fingerprint",
]
