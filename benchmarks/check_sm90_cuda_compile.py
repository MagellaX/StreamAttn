"""Offline-compile StreamAttn SM90 extensions without executing GPU code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def compile_sm90_sources(*, cutlass_root: Path, build_root: Path) -> dict[str, object]:
    import torch
    import torch.utils.cpp_extension as cpp_extension

    from stream_attention.backends.sm90 import tma_pipeline_floor
    from stream_attention.backends.sm90 import transposed_gqa_exact
    from stream_attention.backends.sm90 import micro_prefill_semantics

    build_root.mkdir(parents=True, exist_ok=True)
    components: list[str] = []
    sentinel = object()

    # load_inline compiles before importing the shared object. Replacing only
    # the import step preserves the complete nvcc/ptxas build while allowing it
    # to run on a standard CPU-only GitHub runner.
    with mock.patch.object(
        cpp_extension, "_import_module_from_library", return_value=sentinel
    ):
        transposed_gqa_exact._EXTENSIONS.clear()
        for head_dim in (64, 128):
            extension = transposed_gqa_exact.compile_transposed_gqa_exact_extension(
                cutlass_root=cutlass_root,
                build_dir=build_root / f"exact-d{head_dim}",
                head_dim=head_dim,
                verbose=True,
            )
            if extension is not sentinel:
                raise RuntimeError(f"unexpected exact D{head_dim} compile result")
            components.append(f"exact_d{head_dim}")

        micro_prefill_semantics._EXTENSIONS.clear()
        for head_dim in (64, 128):
            for dtype, causal in (
                (torch.float16, False), (torch.float16, True), (torch.bfloat16, True)
            ):
                extension = micro_prefill_semantics.compile_semantic_extension(
                    head_dim=head_dim, dtype=dtype, causal=causal,
                    cutlass_root=cutlass_root, build_dir=build_root / "micro-semantics",
                    verbose=True,
                )
                if extension is not sentinel:
                    raise RuntimeError("unexpected micro-prefill semantics compile result")
                components.append(f"micro_d{head_dim}_{dtype}_causal{causal}")

        # Each paged build instantiates both layouts and both retained families.
        # The runtime H100 matrix covers the full dtype/mask cross product.
        for head_dim, dtype in ((64, torch.float16), (128, torch.bfloat16)):
            extension = micro_prefill_semantics.compile_semantic_extension(
                head_dim=head_dim, dtype=dtype, causal=True, paged=True,
                cutlass_root=cutlass_root, build_dir=build_root / "micro-paged",
                verbose=True,
            )
            if extension is not sentinel:
                raise RuntimeError("unexpected paged micro-prefill compile result")
            components.append(f"micro_paged_d{head_dim}_{dtype}_causal")

        tma_pipeline_floor._EXTENSIONS.clear()
        extension = tma_pipeline_floor.compile_tma_pipeline_floor_extension(
            cutlass_root=cutlass_root,
            build_dir=build_root / "tma-floor",
            verbose=True,
        )
        if extension is not sentinel:
            raise RuntimeError("unexpected TMA floor compile result")
        components.append("tma_pipeline_floor")

    return {
        "schema": "streamattn.sm90_cuda_compile.v1",
        "passed": True,
        "target": "sm_90a",
        "components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-root", type=Path, default=Path("/tmp/streamattn-sm90-build"))
    args = parser.parse_args()
    result = compile_sm90_sources(
        cutlass_root=args.cutlass_root,
        build_root=args.build_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
