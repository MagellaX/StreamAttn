"""Promoted SM80 exact GQA decode using cp.async and grouped state merging."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .tk_grouped_exact_sources import CPP_SOURCE, CUDA_SOURCE


PROMOTED_EXACT_G8_SPLITS = {
    (1, 32768): 128,
    (2, 32768): 128,
    (4, 16384): 128,
    (4, 32768): 128,
    (4, 65536): 128,
    (8, 32768): 64,
}

PROMOTED_EXACT_G4_SPLITS = {
    (4, 32768): 64,
}

PROMOTED_EXACT_D128_G8_SPLITS = {
    (4, 16384): 64,
}

PROMOTED_EXACT_SHAPES = {
    (16, 2, 8, 64): PROMOTED_EXACT_G8_SPLITS,
    (16, 4, 4, 64): PROMOTED_EXACT_G4_SPLITS,
    (16, 2, 8, 128): PROMOTED_EXACT_D128_G8_SPLITS,
}

_EXTENSIONS: dict[tuple[str, str], Any] = {}
_EXTENSION_LOCK = threading.Lock()


def _tk_candidates(explicit: Optional[Path] = None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    for name in ("STREAMATTN_TK_ROOT", "THUNDERKITTENS_ROOT"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value))
    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            repo_root / "third_party/ThunderKittens",
            repo_root / "artifacts/backend_sources/ThunderKittens",
            Path("/opt/ThunderKittens"),
            Path("/tmp/streamattn_backend_sources/ThunderKittens"),
        ]
    )
    return candidates


def resolve_tk_root(explicit: Optional[Path] = None) -> Path:
    """Resolve a ThunderKittens tree containing ``include/kittens.cuh``."""

    for candidate in _tk_candidates(explicit):
        resolved = candidate.expanduser().resolve()
        if (resolved / "include/kittens.cuh").is_file():
            return resolved
    raise FileNotFoundError(
        "ThunderKittens headers were not found; set STREAMATTN_TK_ROOT to a "
        "tree containing include/kittens.cuh"
    )


def compile_grouped_exact_extension(
    *,
    tk_root: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    verbose: bool = False,
):
    """Compile and cache the validated SM80 grouped exact extension."""

    from torch.utils.cpp_extension import load_inline

    resolved_tk = resolve_tk_root(tk_root)
    if build_dir is None and os.environ.get("STREAMATTN_SM80_EXACT_BUILD_DIR"):
        build_dir = Path(os.environ["STREAMATTN_SM80_EXACT_BUILD_DIR"])
    resolved_build = (
        str(Path(build_dir).expanduser().resolve()) if build_dir is not None else ""
    )
    key = (str(resolved_tk), resolved_build)
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
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
        try:
            extension = load_inline(
                name=f"streamattn_sm80_grouped_exact_{source_id}",
                cpp_sources=CPP_SOURCE,
                cuda_sources=CUDA_SOURCE,
                extra_include_paths=[str(resolved_tk / "include")],
                extra_cflags=["-O3", "-std=c++20"],
                extra_cuda_cflags=[
                    "-O3",
                    "-std=c++20",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-DKITTENS_SM80",
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
    group_size = q_heads // kv_heads if kv_heads > 0 and q_heads % kv_heads == 0 else 0
    if group_size not in (4, 8):
        reasons.append("gqa")
    if dim not in (64, 128):
        reasons.append("head_dim")
    if kv_len <= 0 or kv_len % 16:
        reasons.append("kv_len")
    if not all(t.dtype == torch.bfloat16 for t in (query, key_cache, value_cache)):
        reasons.append("dtype")
    if not all(t.is_contiguous() for t in (query, key_cache, value_cache)):
        reasons.append("layout")
    if promoted_only:
        splits = PROMOTED_EXACT_SHAPES.get((q_heads, kv_heads, group_size, dim))
        if splits is None or (batch, kv_len) not in splits:
            reasons.append("unpromoted_shape")
    return reasons


def supports_grouped_exact(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    promoted_only: bool = True,
    require_tk: bool = True,
) -> bool:
    """Return whether buffers can use the promoted A100 exact specialization."""

    if _shape_reasons(
        query, key_cache, value_cache, promoted_only=promoted_only
    ):
        return False
    if not all(t.is_cuda for t in (query, key_cache, value_cache)):
        return False
    if torch.cuda.get_device_capability(query.device) != (8, 0):
        return False
    if promoted_only and "A100" not in torch.cuda.get_device_name(query.device).upper():
        return False
    if require_tk:
        try:
            resolve_tk_root()
        except FileNotFoundError:
            return False
    return True


@dataclass
class ExactDecodePlan:
    """Allocation-free run plan for the promoted SM80 grouped exact kernel."""

    query: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor
    output: torch.Tensor
    query_flat: torch.Tensor
    output_flat: torch.Tensor
    partial_output: torch.Tensor
    partial_lse: torch.Tensor
    num_chunks: int
    extension: Any
    launch: Any
    backend: str = "sm80_tk_cpasync_grouped_exact"

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        *,
        output: Optional[torch.Tensor] = None,
        num_chunks: Optional[int] = None,
        tk_root: Optional[Path] = None,
        build_dir: Optional[Path] = None,
        compile_verbose: bool = False,
        promoted_only: bool = True,
    ) -> "ExactDecodePlan":
        if not supports_grouped_exact(
            query,
            key_cache,
            value_cache,
            promoted_only=promoted_only,
            require_tk=False,
        ):
            reasons = _shape_reasons(
                query, key_cache, value_cache, promoted_only=promoted_only
            )
            raise ValueError(
                "unsupported SM80 grouped exact decode buffers: "
                + ",".join(reasons or ["device"])
            )
        batch, _, q_heads, dim = map(int, query.shape)
        kv_heads = int(key_cache.shape[1])
        kv_len = int(key_cache.shape[2])
        if num_chunks is None:
            if not promoted_only:
                raise ValueError("num_chunks is required outside the promoted region")
            chunks = PROMOTED_EXACT_SHAPES[
                (q_heads, kv_heads, q_heads // kv_heads, dim)
            ][(batch, kv_len)]
        else:
            chunks = int(num_chunks)
        if chunks <= 0 or chunks % 4 or chunks > kv_len // 16:
            raise ValueError("num_chunks must be divisible by four and fit the KV length")
        if (kv_len // 16) % chunks:
            raise ValueError("num_chunks must divide kv_len/16")
        grouped_chunks = chunks // 4

        if output is None:
            output = torch.empty_like(query)
        if output.shape != query.shape or output.dtype != query.dtype:
            raise ValueError("output must match query shape and dtype")
        if not output.is_cuda or not output.is_contiguous():
            raise ValueError("output must be a contiguous CUDA tensor")

        partial_output = torch.empty(
            batch,
            kv_heads,
            grouped_chunks * 16,
            dim,
            device=query.device,
            dtype=torch.bfloat16,
        )
        partial_lse = torch.empty(
            batch,
            kv_heads,
            grouped_chunks,
            16,
            device=query.device,
            dtype=torch.float32,
        )
        extension = compile_grouped_exact_extension(
            tk_root=tk_root,
            build_dir=build_dir,
            verbose=compile_verbose,
        )
        return cls(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            output=output,
            query_flat=query[:, 0],
            output_flat=output[:, 0],
            partial_output=partial_output,
            partial_lse=partial_lse,
            num_chunks=chunks,
            extension=extension,
            launch=extension.exact_decode_chunk_merged_staged_grouped_direct_out,
        )

    @property
    def workspace_bytes(self) -> int:
        return self.partial_output.numel() * self.partial_output.element_size() + (
            self.partial_lse.numel() * self.partial_lse.element_size()
        )

    def run(self) -> torch.Tensor:
        """Run exact decode into the fixed output buffer."""

        self.launch(
            self.query_flat,
            self.key_cache,
            self.value_cache,
            self.partial_output,
            self.partial_lse,
            self.output_flat,
            self.num_chunks,
        )
        return self.output
