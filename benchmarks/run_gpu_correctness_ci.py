"""Run the contributor GPU correctness suite on an available CUDA device."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch


GENERIC_GPU_TESTS = (
    "tests/test_gate0_projection_bitmask_triton.py",
    "tests/test_gate0_projection_mask_triton.py",
    "tests/test_gate0_projection_scan_triton.py",
    "tests/test_gate0_summary_scan_triton.py",
    "tests/test_qwen_o_proj_triton.py",
    "tests/test_paged_decode.py::test_cuda_paged_exact_matches_dense_and_uses_native_backend",
    "tests/test_seed_only_split_seed.py::test_split_seed_matches_direct_seed_cuda",
    "tests/test_seed_only_route_bundle_decode.py::test_exact_refresh_triton_row_subset_matches_reference_when_cuda_available",
    "tests/test_attention.py::TestStreamAttention::test_fused_online_attention_mask_parity",
    "tests/test_attention.py::TestStreamAttention::test_fused_online_attention_dropout_determinism",
    "tests/test_attention.py::TestStreamAttention::test_fused_online_attention_backward_matches_sdpa",
)


def _device_report() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to report a skipped GPU pass")
    try:
        import triton
    except ImportError as exc:
        raise RuntimeError("Triton is required for the GPU correctness suite") from exc

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "device": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "total_memory_bytes": props.total_memory,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-sm90", action="store_true")
    parser.add_argument("--cutlass-root", type=Path, default=None)
    args = parser.parse_args()

    report = _device_report()
    capability = tuple(report["compute_capability"])
    if args.require_sm90 and capability != (9, 0):
        raise RuntimeError(f"SM90 was required, found compute capability {capability}")

    tests = list(GENERIC_GPU_TESTS)
    if args.require_sm90:
        if args.cutlass_root is None:
            raise RuntimeError("--cutlass-root is required with --require-sm90")
        os.environ["STREAMATTN_CUTLASS_ROOT"] = str(args.cutlass_root.resolve())
        tests.append(
            "tests/test_sm90_exact_backend.py::test_promoted_exact_plan_reuses_workspace_and_tracks_query_mutation"
        )

    print(json.dumps({"schema": "streamattn.gpu_ci_environment.v1", **report}, indent=2))
    command = [sys.executable, "-m", "pytest", "-q", "-rs", *tests]
    print("running:", " ".join(command), flush=True)
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
