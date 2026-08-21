"""Validate GPU source imports and architecture contracts without a GPU."""

from __future__ import annotations

import importlib
import json
from typing import Any


TRITON_MODULES = (
    "stream_attention.core.fused_online_attention",
    "stream_attention.kernels.certified_fwd_triton",
    "stream_attention.kernels.gate0_compact_repair_triton",
    "stream_attention.kernels.gate0_exact_refresh_triton",
    "stream_attention.kernels.gate0_launch_floor_triton",
    "stream_attention.kernels.gate0_projection_bitmask_triton",
    "stream_attention.kernels.gate0_projection_mask_triton",
    "stream_attention.kernels.gate0_projection_scan_triton",
    "stream_attention.kernels.gate0_seed_only_triton",
    "stream_attention.kernels.gate0_summary_scan_triton",
    "stream_attention.kernels.gate1_fwd_triton",
    "stream_attention.kernels.gate1_inline_projection_fwd_triton",
    "stream_attention.kernels.gate1_inline_projection_splitk_triton",
    "stream_attention.kernels.gate1_mass_fwd_triton",
    "stream_attention.kernels.metadata_triton",
    "stream_attention.kernels.metadata_update_triton",
    "stream_attention.kernels.paged_exact_triton",
    "stream_attention.kernels.qwen_o_proj_triton",
)


def check_gpu_source_contract() -> dict[str, Any]:
    failures: list[str] = []
    imported: list[str] = []

    try:
        triton = importlib.import_module("triton")
    except Exception as exc:  # pragma: no cover - exercised by CI failures
        failures.append(f"triton_import:{type(exc).__name__}:{exc}")
        triton_version = None
    else:
        triton_version = getattr(triton, "__version__", "unknown")

    for name in TRITON_MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - exercised by CI failures
            failures.append(f"module_import:{name}:{type(exc).__name__}:{exc}")
            continue
        imported.append(name)
        if hasattr(module, "TRITON_AVAILABLE") and not module.TRITON_AVAILABLE:
            failures.append(f"triton_disabled:{name}")

    try:
        from stream_attention.backends.sm90.transposed_gqa_exact_sources import (
            CUDA_SOURCE,
            cuda_source_for_head_dim,
        )
    except Exception as exc:  # pragma: no cover - exercised by CI failures
        failures.append(f"sm90_source_import:{type(exc).__name__}:{exc}")
    else:
        source_d64 = cuda_source_for_head_dim(64)
        source_d128 = cuda_source_for_head_dim(128)
        if source_d64 is not CUDA_SOURCE:
            failures.append("sm90_d64_source_identity")
        if source_d64 == source_d128:
            failures.append("sm90_d128_source_not_specialized")
        for token in ("wgmma", "sm90_64x8x16", "streamattn_transposed"):
            if token not in source_d64.lower():
                failures.append(f"sm90_source_token_missing:{token}")

    return {
        "schema": "streamattn.gpu_source_contract.v1",
        "passed": not failures,
        "triton_version": triton_version,
        "modules_imported": imported,
        "failures": failures,
    }


def main() -> None:
    result = check_gpu_source_contract()
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
