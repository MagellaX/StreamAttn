"""Calibrate promoted exact kernels into strict universal-exact evidence.

This runner intentionally targets manifest cells rather than arbitrary shape
matrices.  Every eligible external baseline receives a measured or explicit
non-measured outcome, and allocation-bearing eager APIs remain in the artifact
even when a graph-captured, fixed-buffer form is also available.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profile_paged_exact_decode import profile as profile_paged_decode  # noqa: E402
from stream_attention.backends.sm100.gqa_prefill import (  # noqa: E402
    Sm100GqaPrefillPlan,
    TILE_VARIANTS,
)
from stream_attention.benchmark_evidence import (  # noqa: E402
    cuda_environment_fingerprint,
    measured_evidence,
    outcome_evidence,
)
from stream_attention.phase_database import (  # noqa: E402
    BackendEvidence,
    MeasurementStatus,
    write_backend_evidence,
)

try:  # pragma: no cover - availability depends on the GPU PyTorch build
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
    SDPBackend = None
    sdpa_kernel = None


SM90_CELLS: tuple[dict[str, object], ...] = (
    {
        "cell_id": "sm90_decode_b1_32k_g8_d128_hnd",
        "batch": 1,
        "kv_len": 32768,
        "sequence_lengths": [32768],
        "q_heads": 16,
        "kv_heads": 2,
        "layout": "HND",
    },
    {
        "cell_id": "sm90_decode_b8_64k_g4_d128_nhd",
        "batch": 8,
        "kv_len": 65536,
        "sequence_lengths": [65536] * 8,
        "q_heads": 32,
        "kv_heads": 8,
        "layout": "NHD",
    },
    {
        "cell_id": "sm90_decode_b4_tail32k_g8_d128",
        "batch": 4,
        "kv_len": 32768,
        "sequence_lengths": [32767, 32003, 24577, 16385],
        "q_heads": 16,
        "kv_heads": 2,
        "layout": "HND",
    },
    {
        "cell_id": "sm90_decode_b8_tail64k_g4_d128",
        "batch": 8,
        "kv_len": 65536,
        "sequence_lengths": [
            65535,
            65023,
            60001,
            49153,
            32769,
            24577,
            16385,
            8193,
        ],
        "q_heads": 32,
        "kv_heads": 8,
        "layout": "NHD",
    },
)

SM100_CELLS: tuple[dict[str, object], ...] = (
    {"cell_id": "sm100_prefill_b1_256_g8_d128", "batch": 1, "seq_len": 256},
    {"cell_id": "sm100_prefill_b1_384_g8_d128", "batch": 1, "seq_len": 384},
    {"cell_id": "sm100_prefill_b1_512_g8_d128", "batch": 1, "seq_len": 512},
    {"cell_id": "sm100_prefill_b2_128_g8_d128", "batch": 2, "seq_len": 128},
)


def _supported_range(cell_id: str) -> dict[str, object]:
    return {"manifest_id": "universal_exact_v1_20260828", "cell_ids": [cell_id]}


def _nonpaged_baseline_outcomes(
    cell_id: str,
    environment,
) -> Iterable[BackendEvidence]:
    detail = (
        "backend has no direct paged-KV/page-table contract; materializing a "
        "contiguous cache would change the timed workload"
    )
    for backend in ("cudnn_sdpa", "pytorch_sdpa"):
        yield outcome_evidence(
            evidence_id=f"{cell_id}:external:{backend}:direct-paged-unsupported",
            cell_id=cell_id,
            provider="external",
            requested_backend=backend,
            environment=environment,
            status=MeasurementStatus.UNSUPPORTED,
            detail=detail,
        )


def profile_sm90(
    *,
    warmup: int = 10,
    repeats: int = 30,
    paired_trials: int = 9,
    paired_repeats: int = 10,
) -> tuple[BackendEvidence, ...]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("an SM90 H100-class GPU is required")
    environment = cuda_environment_fingerprint(
        library_names=("flashinfer-python", "flashinfer-cubin")
    )
    rows: list[BackendEvidence] = []
    for spec in SM90_CELLS:
        args = argparse.Namespace(
            batch=int(spec["batch"]),
            kv_len=int(spec["kv_len"]),
            q_heads=int(spec["q_heads"]),
            kv_heads=int(spec["kv_heads"]),
            head_dim=128,
            page_size=16,
            layout=str(spec["layout"]),
            dtype="bf16",
            splits=None,
            tokens_per_tile=512,
            partial_num_warps=4,
            sm80_cp_async_experimental=False,
            sm80_grouped_experimental=False,
            sm100_grouped_experimental=False,
            sm100_tgv_experimental=False,
            sm90_fragmented_experimental=True,
            sm90_fragmented_ragged_experimental=True,
            length_profile="manifest",
            sequence_lengths=list(spec["sequence_lengths"]),
            workspace_mb=128,
            flashinfer_backends="fa2,fa3,auto",
            warmup=warmup,
            repeats=repeats,
            paired_trials=paired_trials,
            paired_repeats=paired_repeats,
            atol=0.03,
            seed=17,
        )
        result = profile_paged_decode(args)
        cell_id = str(spec["cell_id"])
        checked_cases = int(spec["batch"]) * int(spec["q_heads"]) * 128
        reference = (
            "paired_exact_crosscheck:"
            f"flashinfer:{result['flashinfer']['resolved_backend']}"
        )
        rows.append(
            measured_evidence(
                evidence_id=f"{cell_id}:streamattn:{result['backend_variant']}",
                cell_id=cell_id,
                provider="streamattn",
                requested_backend="sm90_transposed_gqa_exact_decode",
                resolved_backend=str(result["backend_variant"]),
                environment=environment,
                samples_ms=result["streamattn_samples_ms"],
                correctness_reference=reference,
                checked_cases=checked_cases,
                max_abs_error=float(result["max_abs_error"]),
                max_relative_error=float(result["max_relative_error"]),
                workspace_bytes=int(result["workspace_bytes"]),
                supported_range=_supported_range(cell_id),
                native=True,
                family_id="sm90_transposed_gqa_exact_decode",
                kernel_key=(
                    f"{result['backend_variant']}:s{result['splits']}:"
                    f"t{result['tokens_per_tile']}:w{result['partial_num_warps']}"
                ),
            )
        )
        rows.append(
            measured_evidence(
                evidence_id=(
                    f"{cell_id}:external:flashinfer:"
                    f"{result['flashinfer']['resolved_backend']}"
                ),
                cell_id=cell_id,
                provider="external",
                requested_backend="flashinfer",
                resolved_backend=str(result["flashinfer"]["resolved_backend"]),
                environment=environment,
                samples_ms=result["flashinfer_samples_ms"],
                correctness_reference=reference,
                checked_cases=checked_cases,
                max_abs_error=float(result["max_abs_error"]),
                max_relative_error=float(result["max_relative_error"]),
                workspace_bytes=128 * 1024 * 1024,
                supported_range=_supported_range(cell_id),
                native=False,
            )
        )
        rows.extend(_nonpaged_baseline_outcomes(cell_id, environment))
        print(
            json.dumps(
                {
                    "cell_id": cell_id,
                    "streamattn_ms": result["streamattn_ms"],
                    "flashinfer_ms": result["flashinfer_ms"],
                    "paired_speedup": result["paired"]["speedup_median"],
                    "resolved_flashinfer": result["flashinfer"]["resolved_backend"],
                }
            ),
            flush=True,
        )
    return tuple(rows)


def _time_average_samples(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end) / iterations))
    return samples


def _fp32_causal_gqa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    q = query.float().transpose(1, 2)
    group_size = query.shape[2] // key.shape[2]
    k = key.float().transpose(1, 2).repeat_interleave(group_size, dim=1)
    v = value.float().transpose(1, 2).repeat_interleave(group_size, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    causal = torch.ones(
        query.shape[1], query.shape[1], device=query.device, dtype=torch.bool
    ).tril()
    probabilities = torch.softmax(scores.masked_fill(~causal, -torch.inf), dim=-1)
    return torch.matmul(probabilities, v).transpose(1, 2)


def _errors(output: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    absolute = (output.float() - reference.float()).abs()
    relative = absolute / reference.float().abs().clamp_min(1.0e-6)
    return float(absolute.max().item()), float(relative.max().item())


def _sdpa_function(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    backend,
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        context = (
            contextlib.nullcontext()
            if sdpa_kernel is None or backend is None
            else sdpa_kernel(backend)
        )
        with context:
            return F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                is_causal=True,
                dropout_p=0.0,
                enable_gqa=True,
            ).transpose(1, 2)

    return run


def _graph_capture(
    function: Callable[[], torch.Tensor],
) -> tuple[Callable[[], torch.Tensor], torch.Tensor, int]:
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = function()
    torch.cuda.synchronize()
    workspace = max(0, torch.cuda.memory_allocated() - before)

    def replay() -> torch.Tensor:
        graph.replay()
        return output

    replay()
    torch.cuda.synchronize()
    return replay, output, workspace


def _record_sm100_baseline(
    rows: list[BackendEvidence],
    *,
    cell_id: str,
    backend_name: str,
    resolved_backend: str,
    function: Callable[[], torch.Tensor],
    reference: torch.Tensor,
    environment,
    warmup: int,
    iterations: int,
    repeats: int,
) -> None:
    try:
        eager_output = function()
        torch.cuda.synchronize()
        eager_errors = _errors(eager_output, reference)
        eager_samples = _time_average_samples(
            function,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        rows.append(
            measured_evidence(
                evidence_id=f"{cell_id}:external:{backend_name}:eager-allocating",
                cell_id=cell_id,
                provider="external",
                requested_backend=backend_name,
                resolved_backend=f"{resolved_backend}:eager",
                environment=environment,
                samples_ms=eager_samples,
                correctness_reference="fp32_dense_causal_gqa",
                checked_cases=reference.numel(),
                max_abs_error=eager_errors[0],
                max_relative_error=eager_errors[1],
                workspace_bytes=0,
                supported_range=_supported_range(cell_id),
                native=False,
                correctness_passed=eager_errors[0] <= 0.03,
                failure_reason=(
                    None if eager_errors[0] <= 0.03 else "bf16_tolerance_exceeded"
                ),
                timed_allocation_count=1,
                detail="eager API returns a newly allocated output tensor per call",
            )
        )
        replay, graph_output, graph_workspace = _graph_capture(function)
        graph_errors = _errors(graph_output, reference)
        graph_samples = _time_average_samples(
            replay,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        rows.append(
            measured_evidence(
                evidence_id=f"{cell_id}:external:{backend_name}:cuda-graph",
                cell_id=cell_id,
                provider="external",
                requested_backend=backend_name,
                resolved_backend=f"{resolved_backend}:cuda_graph_replay",
                environment=environment,
                samples_ms=graph_samples,
                correctness_reference="fp32_dense_causal_gqa",
                checked_cases=reference.numel(),
                max_abs_error=graph_errors[0],
                max_relative_error=graph_errors[1],
                workspace_bytes=graph_workspace,
                supported_range=_supported_range(cell_id),
                native=False,
                correctness_passed=graph_errors[0] <= 0.03,
                failure_reason=(
                    None if graph_errors[0] <= 0.03 else "bf16_tolerance_exceeded"
                ),
                timed_allocation_count=0,
                detail="fixed-address CUDA graph replay; capture excluded from timing",
            )
        )
    except Exception as exc:
        torch.cuda.synchronize()
        rows.append(
            outcome_evidence(
                evidence_id=f"{cell_id}:external:{backend_name}:unsupported",
                cell_id=cell_id,
                provider="external",
                requested_backend=backend_name,
                environment=environment,
                status=MeasurementStatus.UNSUPPORTED,
                detail=f"{type(exc).__name__}: {exc}",
            )
        )


def profile_sm100(
    *,
    warmup: int = 5,
    iterations: int = 20,
    repeats: int = 9,
    build_dir: Optional[Path] = None,
) -> tuple[BackendEvidence, ...]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("an SM100a B200-class GPU is required")
    environment = cuda_environment_fingerprint(
        library_names=("flash-attn", "flashinfer-python", "flashinfer-cubin")
    )
    rows: list[BackendEvidence] = []
    for spec in SM100_CELLS:
        cell_id = str(spec["cell_id"])
        batch = int(spec["batch"])
        seq_len = int(spec["seq_len"])
        query = torch.randn(batch, seq_len, 16, 128, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(batch, seq_len, 2, 128, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        reference = _fp32_causal_gqa_reference(query, key, value)
        for tile in TILE_VARIANTS:
            try:
                plan = Sm100GqaPrefillPlan.build(
                    query,
                    key,
                    value,
                    tile=tile,
                    build_dir=build_dir,
                )
                output = plan.run()
                torch.cuda.synchronize()
                errors = _errors(output, reference)
                samples = _time_average_samples(
                    plan.run,
                    warmup=warmup,
                    iterations=iterations,
                    repeats=repeats,
                )
                rows.append(
                    measured_evidence(
                        evidence_id=f"{cell_id}:streamattn:{plan.backend}:{tile}",
                        cell_id=cell_id,
                        provider="streamattn",
                        requested_backend="sm100_tgv_gqa_causal_prefill",
                        resolved_backend=plan.backend,
                        environment=environment,
                        samples_ms=samples,
                        correctness_reference="fp32_dense_causal_gqa",
                        checked_cases=reference.numel(),
                        max_abs_error=errors[0],
                        max_relative_error=errors[1],
                        workspace_bytes=(
                            plan.output.numel() * plan.output.element_size()
                            + plan.sequence_lengths.numel()
                            * plan.sequence_lengths.element_size()
                        ),
                        supported_range=_supported_range(cell_id),
                        native=True,
                        correctness_passed=errors[0] <= 0.03,
                        failure_reason=(
                            None if errors[0] <= 0.03 else "bf16_tolerance_exceeded"
                        ),
                        family_id="sm100_tgv_gqa_causal_prefill",
                        kernel_key=f"{plan.backend}:{tile}",
                    )
                )
            except Exception as exc:
                torch.cuda.synchronize()
                rows.append(
                    outcome_evidence(
                        evidence_id=f"{cell_id}:streamattn:{tile}:error",
                        cell_id=cell_id,
                        provider="streamattn",
                        requested_backend="sm100_tgv_gqa_causal_prefill",
                        environment=environment,
                        status=MeasurementStatus.ERROR,
                        detail=f"{type(exc).__name__}: {exc}",
                        native=True,
                        family_id="sm100_tgv_gqa_causal_prefill",
                    )
                )

        flash_backend = None if SDPBackend is None else SDPBackend.FLASH_ATTENTION
        if flash_backend is None:
            rows.append(
                outcome_evidence(
                    evidence_id=f"{cell_id}:external:pytorch_sdpa:unsupported",
                    cell_id=cell_id,
                    provider="external",
                    requested_backend="pytorch_sdpa",
                    environment=environment,
                    status=MeasurementStatus.UNSUPPORTED,
                    detail="PyTorch build does not expose forced Flash SDPA",
                )
            )
        else:
            _record_sm100_baseline(
                rows,
                cell_id=cell_id,
                backend_name="pytorch_sdpa",
                resolved_backend="torch_sdpa_flash_attention",
                function=_sdpa_function(query, key, value, flash_backend),
                reference=reference,
                environment=environment,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )
        cudnn_backend = (
            None if SDPBackend is None else getattr(SDPBackend, "CUDNN_ATTENTION", None)
        )
        if cudnn_backend is None:
            rows.append(
                outcome_evidence(
                    evidence_id=f"{cell_id}:external:cudnn_sdpa:unsupported",
                    cell_id=cell_id,
                    provider="external",
                    requested_backend="cudnn_sdpa",
                    environment=environment,
                    status=MeasurementStatus.UNSUPPORTED,
                    detail="PyTorch build does not expose forced cuDNN SDPA",
                )
            )
        else:
            _record_sm100_baseline(
                rows,
                cell_id=cell_id,
                backend_name="cudnn_sdpa",
                resolved_backend="torch_sdpa_cudnn_attention",
                function=_sdpa_function(query, key, value, cudnn_backend),
                reference=reference,
                environment=environment,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )

        try:
            from flash_attn import flash_attn_func

            def flashattention_fn() -> torch.Tensor:
                return flash_attn_func(query, key, value, causal=True)
            _record_sm100_baseline(
                rows,
                cell_id=cell_id,
                backend_name="flashattention",
                resolved_backend="flash_attn_func",
                function=flashattention_fn,
                reference=reference,
                environment=environment,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )
        except Exception as exc:
            rows.append(
                outcome_evidence(
                    evidence_id=f"{cell_id}:external:flashattention:unsupported",
                    cell_id=cell_id,
                    provider="external",
                    requested_backend="flashattention",
                    environment=environment,
                    status=MeasurementStatus.UNSUPPORTED,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )

        if batch != 1:
            rows.append(
                outcome_evidence(
                    evidence_id=f"{cell_id}:external:flashinfer_prefill:unsupported",
                    cell_id=cell_id,
                    provider="external",
                    requested_backend="flashinfer_prefill",
                    environment=environment,
                    status=MeasurementStatus.UNSUPPORTED,
                    detail="single-prefill API does not implement batched GQA semantics",
                )
            )
        else:
            try:
                import flashinfer

                def flashinfer_fn() -> torch.Tensor:
                    return flashinfer.single_prefill_with_kv_cache(
                        query[0], key[0], value[0], causal=True, kv_layout="NHD"
                    ).unsqueeze(0)
                _record_sm100_baseline(
                    rows,
                    cell_id=cell_id,
                    backend_name="flashinfer_prefill",
                    resolved_backend="flashinfer_single_prefill",
                    function=flashinfer_fn,
                    reference=reference,
                    environment=environment,
                    warmup=warmup,
                    iterations=iterations,
                    repeats=repeats,
                )
            except Exception as exc:
                rows.append(
                    outcome_evidence(
                        evidence_id=(
                            f"{cell_id}:external:flashinfer_prefill:unsupported"
                        ),
                        cell_id=cell_id,
                        provider="external",
                        requested_backend="flashinfer_prefill",
                        environment=environment,
                        status=MeasurementStatus.UNSUPPORTED,
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
        print(json.dumps({"cell_id": cell_id, "evidence_rows": len(rows)}), flush=True)
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("sm90", "sm100"), required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    if args.architecture == "sm90":
        evidence = profile_sm90(warmup=args.warmup, repeats=args.repeats)
    else:
        evidence = profile_sm100(
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            build_dir=args.build_dir,
        )
    write_backend_evidence(evidence, args.output_json)
    print(json.dumps({"artifact": str(args.output_json), "rows": len(evidence)}))


if __name__ == "__main__":
    main()
