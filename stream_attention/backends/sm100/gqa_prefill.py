"""Native Blackwell contiguous causal GQA prefill experiment."""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .gqa_prefill_sources import CPP_SOURCE, CUDA_SOURCE
from .paged_gqa_exact import resolve_sm100_cutlass_root


_EXTENSIONS: dict[tuple[str, str], Any] = {}
_EXTENSION_LOCK = threading.Lock()

TILE_VARIANTS = {
    "h8_q1": 0,
    "h8_q2": 1,
    "h8_q4": 2,
}

PROMOTED_SM100_GQA_PREFILL_CELLS = frozenset(
    {
        (1, 64),
        (1, 128),
        (1, 256),
        (1, 384),
        (2, 64),
    }
)


def compile_sm100_gqa_prefill_extension(
    *,
    cutlass_root: Optional[Path] = None,
    build_dir: Optional[Path] = None,
    verbose: bool = False,
):
    """Compile and cache the SM100 contiguous GQA prefill extension."""

    from torch.utils.cpp_extension import load_inline

    resolved_cutlass = resolve_sm100_cutlass_root(cutlass_root)
    if build_dir is None and os.environ.get("STREAMATTN_SM100_BUILD_DIR"):
        build_dir = Path(os.environ["STREAMATTN_SM100_BUILD_DIR"])
    resolved_build = (
        str(Path(build_dir).expanduser().resolve()) if build_dir is not None else ""
    )
    csrc = Path(__file__).resolve().parent / "csrc"
    header_bytes = b"".join(
        (csrc / name).read_bytes() for name in ("common.cuh", "tgv_gqa.cuh")
    )
    key = (str(resolved_cutlass), resolved_build)
    with _EXTENSION_LOCK:
        cached = _EXTENSIONS.get(key)
        if cached is not None:
            return cached

        source_id = hashlib.sha1(
            CPP_SOURCE.encode("utf-8")
            + CUDA_SOURCE.encode("utf-8")
            + header_bytes
            + key[0].encode("utf-8")
        ).hexdigest()[:12]
        kwargs: dict[str, Any] = {}
        if build_dir is not None:
            resolved_build_path = Path(resolved_build)
            resolved_build_path.mkdir(parents=True, exist_ok=True)
            kwargs["build_directory"] = str(resolved_build_path)

        previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
        try:
            extension = load_inline(
                name=f"streamattn_sm100_gqa_prefill_{source_id}",
                cpp_sources=CPP_SOURCE,
                cuda_sources=CUDA_SOURCE,
                extra_include_paths=[
                    str(csrc),
                    str(resolved_cutlass / "include"),
                ],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=[
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
                    "-gencode=arch=compute_100a,code=sm_100a",
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


def supports_sm100_gqa_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    """Return whether tensors match the first native B200 prefill scope."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if key.shape != value.shape:
        return False
    if query.shape[:2] != key.shape[:2]:
        return False
    if tuple(query.shape[2:]) != (16, 128):
        return False
    if tuple(key.shape[2:]) != (2, 128):
        return False
    if not all(t.dtype == torch.bfloat16 for t in (query, key, value)):
        return False
    if not all(t.is_cuda and t.is_contiguous() for t in (query, key, value)):
        return False
    if not (query.device == key.device == value.device):
        return False
    return torch.cuda.get_device_capability(query.device) == (10, 0)


def is_promoted_sm100_gqa_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    """Return whether a shape is inside the measured profitable B200 phase."""

    if not supports_sm100_gqa_prefill(query, key, value):
        return False
    return (int(query.shape[0]), int(query.shape[1])) in (
        PROMOTED_SM100_GQA_PREFILL_CELLS
    )


@dataclass
class Sm100GqaPrefillPlan:
    """Allocation-free run plan for native B200 causal grouped prefill."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    sequence_lengths: torch.Tensor
    tile_variant: int
    extension: Any
    launch: Any
    backend: str = "sm100_tgv_gqa_causal_prefill"

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        tile: str = "h8_q4",
        output: Optional[torch.Tensor] = None,
        cutlass_root: Optional[Path] = None,
        build_dir: Optional[Path] = None,
        compile_verbose: bool = False,
    ) -> "Sm100GqaPrefillPlan":
        if not supports_sm100_gqa_prefill(query, key, value):
            raise ValueError("unsupported SM100 contiguous GQA prefill tensors")
        try:
            tile_variant = TILE_VARIANTS[tile]
        except KeyError as exc:
            raise ValueError(f"unsupported tile {tile!r}") from exc
        if output is None:
            output = torch.empty_like(query)
        if (
            output.shape != query.shape
            or output.dtype != query.dtype
            or output.device != query.device
            or not output.is_contiguous()
        ):
            raise ValueError("output must be a contiguous tensor matching query")
        sequence_lengths = torch.full(
            (query.shape[0],),
            int(query.shape[1]),
            device=query.device,
            dtype=torch.int32,
        )
        extension = compile_sm100_gqa_prefill_extension(
            cutlass_root=cutlass_root,
            build_dir=build_dir,
            verbose=compile_verbose,
        )
        return cls(
            query=query,
            key=key,
            value=value,
            output=output,
            sequence_lengths=sequence_lengths,
            tile_variant=tile_variant,
            extension=extension,
            launch=extension.prefill_out,
        )

    def run(self) -> torch.Tensor:
        self.launch(
            self.query,
            self.key,
            self.value,
            self.sequence_lengths,
            self.output,
            self.tile_variant,
        )
        return self.output


__all__ = [
    "PROMOTED_SM100_GQA_PREFILL_CELLS",
    "Sm100GqaPrefillPlan",
    "TILE_VARIANTS",
    "compile_sm100_gqa_prefill_extension",
    "is_promoted_sm100_gqa_prefill",
    "supports_sm100_gqa_prefill",
]
