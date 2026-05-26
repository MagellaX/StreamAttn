"""Analytical seed-only kernel mode autotuner.

This script encodes the route math behind the current StreamAttn wedge:

* head-private seed-only can duplicate tiny seed K/V reads across Q heads when
  ``G * S / N`` remains small;
* split-seed mode increases CTA count for batch sizes below the calibrated
  direct-kernel occupancy threshold;
* GQA-shared seed mode preserves bytes but can under-supply CTAs when Hkv is
  small.

It does not time kernels by itself.  Use it to choose which CUDA/TK seed-only
kernel family to benchmark next.  The default H100 threshold is calibrated from
the planned-direct B4 evidence; pass a larger ``--target-waves`` value when you
want conservative split-seed diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stream_attention import (  # noqa: E402
    Gate0SeedOnlyBatchedPolicy,
    autotune_seed_kernel_mode,
    seed_shape_from_policy,
)


def _parse_ints(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"empty integer list: {raw!r}")
    return values


def _dtype_bytes(dtype: str) -> int:
    if dtype in {"fp16", "bf16"}:
        return 2
    if dtype in {"fp32"}:
        return 4
    if dtype in {"fp8", "int8"}:
        return 1
    raise ValueError(f"unsupported dtype for byte model: {dtype}")


def profile(args: argparse.Namespace) -> dict:
    policy = Gate0SeedOnlyBatchedPolicy.from_json(args.policy)
    rows = []
    for batch in _parse_ints(args.batch_sizes):
        shape = seed_shape_from_policy(
            policy,
            batch=batch,
            dtype_bytes=_dtype_bytes(args.dtype),
        )
        result = autotune_seed_kernel_mode(
            shape,
            sm_count=args.sm_count,
            target_waves=args.target_waves,
            seed_tile_tokens=_parse_ints(args.seed_tile_tokens),
            duplication_byte_budget=args.duplication_byte_budget,
        )
        rows.append(
            {
                "batch": batch,
                "shape": result.shape.to_dict(),
                "decision": result.decision,
                "recommended_mode": result.recommended_mode,
                "recommended_seed_tile_tokens": result.recommended_seed_tile_tokens,
                "top_candidates": [
                    candidate.to_dict()
                    for candidate in result.candidates[: args.top_k]
                ],
            }
        )

    first_direct_viable = next(
        (
            row["batch"]
            for row in rows
            if row["recommended_mode"] == "head_private_direct_seed"
            and row["decision"] == "seed_only_native_candidate"
        ),
        None,
    )
    first_seed_viable = next(
        (
            row["batch"]
            for row in rows
            if row["decision"] == "seed_only_native_candidate"
        ),
        None,
    )
    return {
        "schema": "streamattn.seed_kernel_mode_autotune.profile.v1",
        "policy_id": policy.policy_id,
        "policy_path": str(args.policy),
        "dtype": args.dtype,
        "sm_count": args.sm_count,
        "target_waves": args.target_waves,
        "duplication_byte_budget": args.duplication_byte_budget,
        "seed_tile_tokens": _parse_ints(args.seed_tile_tokens),
        "first_seed_viable_batch": first_seed_viable,
        "first_direct_viable_batch": first_direct_viable,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("stream_attention/policies/qwen25_05b_l8_32k_seed_only_batched.json"),
    )
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32", "fp8", "int8"])
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--sm-count", type=int, default=132)
    parser.add_argument("--target-waves", type=float, default=0.40)
    parser.add_argument("--duplication-byte-budget", type=float, default=0.15)
    parser.add_argument("--seed-tile-tokens", default="384,256,192,128,96,64,32")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(profile(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
