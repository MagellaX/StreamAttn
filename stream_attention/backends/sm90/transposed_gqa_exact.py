"""Planned Hopper exact decode using the transposed true-GQA WGMMA dataflow.

The production specialization deliberately consumes head-major contiguous KV
buffers. Converting BNHD caches in ``run()`` would erase the narrow exact-kernel
win, so cache layout conversion remains an explicit model-adapter concern.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .transposed_gqa_exact_sources import CPP_SOURCE, CUDA_SOURCE


PROMOTED_EXACT_SHAPE = {
    "q_heads": 16,
    "kv_heads": 2,
    "group_size": 8,
    "head_dim": 64,
    "dtype": torch.bfloat16,
}

# Calibrated with paired FlashInfer gates on H100 80GB. Keep this discrete:
# nearby cells can cross parity because producer waves and tiles/split are
# quantized, while the split-state merge grows with the split count.
PROMOTED_EXACT_SPLITS = {
    (2, 16384): 64,
    (4, 16384): 64,
    (4, 32768): 64,
    (4, 65536): 64,
    (8, 16384): 32,
    (8, 32768): 32,
    (8, 65536): 32,
}

_EXTENSIONS: dict[tuple[str, str], Any] = {}
_EXTENSION_LOCK = threading.Lock()


def choose_num_splits(
    *,
    batch: int,
    kv_heads: int,
    kv_len: int,
    target_producer_ctas: int = 256,
) -> int:
    """Choose the smallest split count that reaches the producer CTA target."""

    if batch <= 0 or kv_heads <= 0 or kv_len <= 0 or kv_len % 64:
        raise ValueError("batch/kv_heads must be positive and kv_len divisible by 64")
    num_tiles = kv_len // 64
    needed = (target_producer_ctas + batch * kv_heads - 1) // (batch * kv_heads)
    return max(1, min(num_tiles, needed))


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
    """Resolve a CUTLASS tree containing the CUTE headers used by the kernel."""

    for candidate in _cutlass_candidates(explicit):
        resolved = candidate.expanduser().resolve()
        if (resolved / "include/cute/tensor.hpp").is_file():
            return resolved
    raise FileNotFoundError(
        "CUTLASS headers were not found; set STREAMATTN_CUTLASS_ROOT to a "
        "CUTLASS tree containing include/cute/tensor.hpp"
    )


def compile_transposed_gqa_exact_extension(
    *,
    cutlass_root: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    verbose: bool = False,
):
    """Compile once during planning and cache the loaded SM90 extension."""

    from torch.utils.cpp_extension import load_inline

    resolved_cutlass = resolve_cutlass_root(cutlass_root)
    if build_dir is None and os.environ.get("STREAMATTN_EXACT_BUILD_DIR"):
        build_dir = Path(os.environ["STREAMATTN_EXACT_BUILD_DIR"])
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
            resolved_build_path = Path(resolved_build)
            resolved_build_path.mkdir(parents=True, exist_ok=True)
            kwargs["build_directory"] = str(resolved_build_path)

        previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
        try:
            extension = load_inline(
                name=f"streamattn_sm90_exact_{source_id}",
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


def _shape_reasons(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    promoted_only: bool,
) -> list[str]:
    reasons: list[str] = []
    if query.dim() != 4 or key_cache.dim() != 4 or value_cache.dim() != 4:
        return ["rank"]
    if query.shape[1] != 1:
        reasons.append("query_len")
    if key_cache.shape != value_cache.shape:
        reasons.append("kv_shape")
        return reasons
    batch, _, q_heads, dim = map(int, query.shape)
    kv_batch, kv_heads, kv_len, kv_dim = map(int, key_cache.shape)
    if batch != kv_batch or dim != kv_dim:
        reasons.append("shape")
    if kv_heads <= 0 or q_heads % kv_heads or q_heads // kv_heads != 8:
        reasons.append("gqa")
    if dim != 64:
        reasons.append("head_dim")
    if kv_len <= 0 or kv_len % 64:
        reasons.append("kv_len")
    if not all(t.dtype == torch.bfloat16 for t in (query, key_cache, value_cache)):
        reasons.append("dtype")
    if not all(t.is_contiguous() for t in (query, key_cache, value_cache)):
        reasons.append("layout")
    if promoted_only:
        expected = PROMOTED_EXACT_SHAPE
        if (
            q_heads != expected["q_heads"]
            or kv_heads != expected["kv_heads"]
            or (batch, kv_len) not in PROMOTED_EXACT_SPLITS
        ):
            reasons.append("unpromoted_shape")
    return reasons


def supports_transposed_gqa_exact(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    promoted_only: bool = True,
    require_cutlass: bool = True,
) -> bool:
    """Return whether buffers can use the promoted SM90 specialization."""

    if _shape_reasons(
        query, key_cache, value_cache, promoted_only=promoted_only
    ):
        return False
    if not all(t.is_cuda for t in (query, key_cache, value_cache)):
        return False
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        return False
    if require_cutlass:
        try:
            resolve_cutlass_root()
        except FileNotFoundError:
            return False
    return True


@dataclass
class ExactDecodePlan:
    """Allocation-free run plan for the transposed SM90 exact decode kernel."""

    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    output: torch.Tensor
    query_group: torch.Tensor
    output_group: torch.Tensor
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    num_splits: int
    extension: Any
    partial_launch: Any
    merge_launch: Any
    warp_merge_launch: Any
    combined_launch: Any
    backend: str = "sm90_transposed_gqa_wgmma_exact"

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        output: Optional[torch.Tensor] = None,
        num_splits: Optional[int] = None,
        cutlass_root: Optional[Path] = None,
        build_dir: Optional[Path] = None,
        compile_verbose: bool = False,
        promoted_only: bool = True,
    ) -> "ExactDecodePlan":
        if not supports_transposed_gqa_exact(
            query,
            key_cache,
            value_cache,
            promoted_only=promoted_only,
            require_cutlass=False,
        ):
            reasons = _shape_reasons(
                query, key_cache, value_cache, promoted_only=promoted_only
            )
            raise ValueError(
                "unsupported SM90 transposed exact decode buffers: "
                + ",".join(reasons or ["device"])
            )
        batch, _, q_heads, dim = map(int, query.shape)
        kv_heads = int(key_cache.shape[1])
        kv_len = int(key_cache.shape[2])
        if num_splits is not None:
            splits = num_splits
        elif promoted_only:
            splits = PROMOTED_EXACT_SPLITS[(batch, kv_len)]
        else:
            splits = choose_num_splits(
                batch=batch, kv_heads=kv_heads, kv_len=kv_len
            )
        if splits <= 0 or splits > kv_len // 64:
            raise ValueError("num_splits must be in [1, kv_len/64]")
        if output is None:
            output = torch.empty_like(query)
        if output.shape != query.shape or output.dtype != query.dtype:
            raise ValueError("output must match query shape and dtype")
        if not output.is_cuda or not output.is_contiguous():
            raise ValueError("output must be a contiguous CUDA tensor")

        groups = batch * kv_heads
        partial_output = torch.empty(
            groups,
            splits,
            8,
            dim,
            device=query.device,
            dtype=torch.float32,
        )
        partial_lse = torch.empty(
            groups,
            splits,
            8,
            device=query.device,
            dtype=torch.float32,
        )
        extension = compile_transposed_gqa_exact_extension(
            cutlass_root=cutlass_root,
            build_dir=build_dir,
            verbose=compile_verbose,
        )
        return cls(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            query_group=query.view(batch, kv_heads, q_heads // kv_heads, dim),
            output_group=output.view(
                batch * kv_heads, q_heads // kv_heads, dim
            ),
            partial_output=partial_output,
            partial_lse=partial_lse,
            num_splits=splits,
            extension=extension,
            partial_launch=extension.exact_partial_out,
            merge_launch=extension.exact_merge_out,
            warp_merge_launch=extension.exact_merge_warp_out,
            combined_launch=extension.exact_decode_out,
        )

    @property
    def workspace_bytes(self) -> int:
        return self.partial_output.numel() * self.partial_output.element_size() + (
            self.partial_lse.numel() * self.partial_lse.element_size()
        )

    def run_two_call(self) -> torch.Tensor:
        """Launch producer and merge through separate extension dispatches."""

        self.partial_launch(
            self.query_group,
            self.key_cache,
            self.value_cache,
            self.partial_output,
            self.partial_lse,
            self.num_splits,
        )
        self.merge_launch(
            self.partial_output, self.partial_lse, self.output_group
        )
        return self.output

    def run_combined(self) -> torch.Tensor:
        """Launch producer and merge through one extension dispatch."""

        self.combined_launch(
            self.query_group,
            self.key_cache,
            self.value_cache,
            self.partial_output,
            self.partial_lse,
            self.output_group,
            self.num_splits,
        )
        return self.output

    def run_warp_merge(self) -> torch.Tensor:
        """Launch the producer followed by the promoted one-warp merge."""

        self.partial_launch(
            self.query_group,
            self.key_cache,
            self.value_cache,
            self.partial_output,
            self.partial_lse,
            self.num_splits,
        )
        self.warp_merge_launch(
            self.partial_output, self.partial_lse, self.output_group
        )
        return self.output

    def run(self) -> torch.Tensor:
        """Run the H100-promoted producer and one-warp merge path."""

        return self.run_warp_merge()
