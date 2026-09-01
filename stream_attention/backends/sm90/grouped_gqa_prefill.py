"""Experimental natural-orientation Hopper exact GQA prefill backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from .transposed_gqa_exact import compile_transposed_gqa_exact_extension


RESOURCE_FIELDS = (
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "blocks_per_sm",
    "max_threads_per_block",
)


def supports_grouped_wgmma_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> bool:
    """Return whether tensors match the first SM90 grouped-prefill canary."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        return False
    if key.shape != value.shape:
        return False
    if query.shape[:2] != key.shape[:2] or query.shape[3] != key.shape[3]:
        return False
    q_heads = int(query.shape[2])
    kv_heads = int(key.shape[2])
    if kv_heads <= 0 or q_heads % kv_heads:
        return False
    if q_heads // kv_heads not in (4, 8) or int(query.shape[3]) != 128:
        return False
    if not all(t.dtype == torch.bfloat16 for t in (query, key, value)):
        return False
    if not all(t.is_cuda and t.is_contiguous() for t in (query, key, value)):
        return False
    if not (query.device == key.device == value.device):
        return False
    return torch.cuda.get_device_capability(query.device) == (9, 0)


def decode_grouped_prefill_resources(values: torch.Tensor) -> dict[str, int]:
    """Decode the compact CUDA resource vector returned by the extension."""

    raw = [int(value) for value in values.cpu().tolist()]
    if len(raw) != len(RESOURCE_FIELDS):
        raise ValueError(
            f"expected {len(RESOURCE_FIELDS)} resource values, got {len(raw)}"
        )
    return dict(zip(RESOURCE_FIELDS, raw))


@dataclass
class GroupedWgmmaPrefillPlan:
    """Allocation-free plan for the experimental SM90 grouped-prefill kernel."""

    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    extension: Any
    launch: Any
    backend: str = "sm90_grouped_m64n64x2_wgmma_prefill_canary"

    @classmethod
    def build(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        output: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        cutlass_root: Optional[Path] = None,
        build_dir: Optional[Path] = None,
        compile_verbose: bool = False,
    ) -> "GroupedWgmmaPrefillPlan":
        if not supports_grouped_wgmma_prefill(query, key, value):
            raise ValueError("unsupported SM90 grouped-prefill tensors")
        if output is None:
            output = torch.empty_like(query)
        if (
            output.shape != query.shape
            or output.dtype != query.dtype
            or output.device != query.device
            or not output.is_contiguous()
        ):
            raise ValueError("output must be a contiguous tensor matching query")
        expected_lse = (query.shape[0], query.shape[1], query.shape[2])
        if lse is None:
            lse = torch.empty(expected_lse, device=query.device, dtype=torch.float32)
        if (
            tuple(lse.shape) != expected_lse
            or lse.dtype != torch.float32
            or lse.device != query.device
            or not lse.is_contiguous()
        ):
            raise ValueError("lse must be contiguous FP32 [B,S,Hq]")
        extension = compile_transposed_gqa_exact_extension(
            cutlass_root=cutlass_root,
            build_dir=build_dir,
            head_dim=int(query.shape[-1]),
            verbose=compile_verbose,
        )
        return cls(
            query=query,
            key=key,
            value=value,
            output=output,
            lse=lse,
            extension=extension,
            launch=extension.grouped_wgmma_prefill_out,
        )

    def run(self) -> torch.Tensor:
        self.launch(self.query, self.key, self.value, self.output, self.lse)
        return self.output

    def resource_info(self) -> dict[str, int]:
        return decode_grouped_prefill_resources(
            self.extension.grouped_wgmma_prefill_resource_info()
        )


__all__ = [
    "GroupedWgmmaPrefillPlan",
    "RESOURCE_FIELDS",
    "decode_grouped_prefill_resources",
    "supports_grouped_wgmma_prefill",
]
