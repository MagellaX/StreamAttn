"""Offline-compile StreamAttn SM90 extensions without executing GPU code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest import mock


def compile_sm90_sources(*, cutlass_root: Path, build_root: Path) -> dict[str, object]:
    import torch.utils.cpp_extension as cpp_extension

    from stream_attention.backends.sm90 import tma_pipeline_floor
    from stream_attention.backends.sm90 import transposed_gqa_exact

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
