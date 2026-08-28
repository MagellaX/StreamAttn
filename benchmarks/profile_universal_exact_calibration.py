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
from stream_attention.core.fused_online_attention import (  # noqa: E402
    FusedOnlineAttention,
)
from stream_attention.benchmark_evidence import (  # noqa: E402
    cuda_environment_fingerprint,
    measured_evidence,
    outcome_evidence,
)
from stream_attention.kernels.grouped_gqa_prefill_triton import (  # noqa: E402
    grouped_gqa_prefill,
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


SM80_CELLS: tuple[dict[str, object], ...] = (
    {
        "cell_id": "sm80_decode_b1_32k_g8_d128_hnd",
        "batch": 1,
        "kv_len": 32768,
        "sequence_lengths": [32768],
        "q_heads": 16,
        "kv_heads": 2,
        "head_dim": 128,
        "dtype": "bf16",
        "layout": "HND",
    },
    {
        "cell_id": "sm80_decode_b4_tail16k_g8_d64",
        "batch": 4,
        # Physical paged capacity is rounded up from the 16,385-token tail.
        "kv_len": 16400,
        "sequence_lengths": [16385, 16271, 16001, 8193],
        "q_heads": 16,
        "kv_heads": 2,
        "head_dim": 64,
        "dtype": "fp16",
        "layout": "NHD",
    },
)


SM80_ADDITIONAL_PREFILL_CELLS: tuple[dict[str, object], ...] = (
    {
        "cell_id": "sm80_prefill_b1_tail257_g4_d128",
        "batch": 1,
        "seq_len": 257,
        "q_heads": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "dtype": torch.bfloat16,
        "causal": True,
        "layout": "NHD",
    },
    {
        "cell_id": "sm80_prefill_noncausal_mha_d64",
        "batch": 2,
        "seq_len": 1024,
        "q_heads": 16,
        "kv_heads": 16,
        "head_dim": 64,
        "dtype": torch.float16,
        "causal": False,
        "layout": "HND",
    },
)


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


def _profile_paged_cells(
    *,
    cells: tuple[dict[str, object], ...],
    capability: tuple[int, int],
    gpu_description: str,
    native_backend: str,
    family_id: str,
    flashinfer_backends: str,
    sm80_cp_async: bool,
    sm90_fragmented: bool,
    warmup: int = 10,
    repeats: int = 30,
    paired_trials: int = 9,
    paired_repeats: int = 10,
) -> tuple[BackendEvidence, ...]:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != capability:
        raise RuntimeError(f"{gpu_description} is required")
    environment = cuda_environment_fingerprint(
        library_names=("flash-attn", "flashinfer-python", "flashinfer-cubin")
    )
    rows: list[BackendEvidence] = []
    for spec in cells:
        head_dim = int(spec.get("head_dim", 128))
        dtype = str(spec.get("dtype", "bf16"))
        tolerance = 0.015 if dtype == "fp16" else 0.03
        args = argparse.Namespace(
            batch=int(spec["batch"]),
            kv_len=int(spec["kv_len"]),
            q_heads=int(spec["q_heads"]),
            kv_heads=int(spec["kv_heads"]),
            head_dim=head_dim,
            page_size=16,
            layout=str(spec["layout"]),
            dtype=dtype,
            splits=None,
            tokens_per_tile=512,
            partial_num_warps=4,
            sm80_cp_async_experimental=sm80_cp_async,
            sm80_grouped_experimental=False,
            sm100_grouped_experimental=False,
            sm100_tgv_experimental=False,
            sm90_fragmented_experimental=sm90_fragmented,
            sm90_fragmented_ragged_experimental=sm90_fragmented,
            length_profile="manifest",
            sequence_lengths=list(spec["sequence_lengths"]),
            workspace_mb=128,
            flashinfer_backends=flashinfer_backends,
            warmup=warmup,
            repeats=repeats,
            paired_trials=paired_trials,
            paired_repeats=paired_repeats,
            atol=tolerance,
            seed=17,
        )
        result = profile_paged_decode(args)
        cell_id = str(spec["cell_id"])
        checked_cases = int(spec["batch"]) * int(spec["q_heads"]) * head_dim
        reference = (
            "paired_exact_crosscheck:"
            f"flashinfer:{result['flashinfer']['resolved_backend']}"
        )
        resolved_family = family_id
        source_suffix = ""
        if str(result["backend_variant"]) == "streamattn_paged_exact_native":
            resolved_family = "triton_paged_exact_decode"
        elif str(result["backend_variant"]) == "streamattn_paged_sm80_cp_async_exact":
            from stream_attention.backends.sm80.paged_gqa_exact import (
                sm80_paged_gqa_source_id,
            )

            source_suffix = f":src{sm80_paged_gqa_source_id()}"
        rows.append(
            measured_evidence(
                evidence_id=f"{cell_id}:streamattn:{result['backend_variant']}",
                cell_id=cell_id,
                provider="streamattn",
                requested_backend=native_backend,
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
                family_id=resolved_family,
                kernel_key=(
                    f"{result['backend_variant']}:s{result['splits']}:"
                    f"t{result['tokens_per_tile']}:w{result['partial_num_warps']}"
                    f"{source_suffix}"
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


def profile_sm80(
    *,
    warmup: int = 10,
    repeats: int = 30,
    paired_trials: int = 9,
    paired_repeats: int = 10,
) -> tuple[BackendEvidence, ...]:
    decode_rows = profile_sm80_decode_surface(
        warmup=warmup,
        repeats=repeats,
        paired_trials=paired_trials,
        paired_repeats=paired_repeats,
    )
    prefill_rows = profile_sm80_prefill_surface(
        warmup=warmup,
        repeats=repeats,
        paired_repeats=paired_repeats,
    )
    dropout_rows = _profile_sm80_training_dropout(
        warmup=max(3, min(warmup, 5)),
        iterations=max(3, min(paired_repeats, 5)),
        repeats=max(5, min(repeats, 9)),
    )
    training_rows = _profile_sm80_training(
        warmup=max(3, min(warmup, 5)),
        iterations=max(3, min(paired_repeats, 5)),
        repeats=max(5, min(repeats, 9)),
    )
    return decode_rows + prefill_rows + dropout_rows + training_rows


def profile_sm80_decode_surface(
    *,
    warmup: int = 10,
    repeats: int = 30,
    paired_trials: int = 9,
    paired_repeats: int = 10,
) -> tuple[BackendEvidence, ...]:
    return _profile_paged_cells(
        cells=SM80_CELLS,
        capability=(8, 0),
        gpu_description="an SM80 A100-class GPU",
        native_backend="sm80_cp_async_gqa_exact_decode",
        family_id="sm80_paged_gqa_exact_decode",
        flashinfer_backends="fa2,auto",
        sm80_cp_async=True,
        sm90_fragmented=False,
        warmup=warmup,
        repeats=repeats,
        paired_trials=paired_trials,
        paired_repeats=paired_repeats,
    )


def profile_sm80_prefill_surface(
    *,
    warmup: int = 10,
    repeats: int = 30,
    paired_repeats: int = 10,
) -> tuple[BackendEvidence, ...]:
    prefill_rows = _profile_sm80_prefill(
        warmup=warmup,
        iterations=max(10, paired_repeats),
        repeats=max(9, min(repeats, 15)),
    )
    tail_prefill_rows = _profile_sm80_tail_prefill(
        warmup=warmup,
        iterations=max(10, paired_repeats),
        repeats=max(9, min(repeats, 15)),
    )
    hnd_prefill_rows = _profile_sm80_hnd_noncausal_prefill(
        warmup=warmup,
        iterations=max(10, paired_repeats),
        repeats=max(9, min(repeats, 15)),
    )
    return prefill_rows + tail_prefill_rows + hnd_prefill_rows


def profile_sm90(
    *,
    warmup: int = 10,
    repeats: int = 30,
    paired_trials: int = 9,
    paired_repeats: int = 10,
) -> tuple[BackendEvidence, ...]:
    return _profile_paged_cells(
        cells=SM90_CELLS,
        capability=(9, 0),
        gpu_description="an SM90 H100-class GPU",
        native_backend="sm90_transposed_gqa_exact_decode",
        family_id="sm90_transposed_gqa_exact_decode",
        flashinfer_backends="fa2,fa3,auto",
        sm80_cp_async=False,
        sm90_fragmented=True,
        warmup=warmup,
        repeats=repeats,
        paired_trials=paired_trials,
        paired_repeats=paired_repeats,
    )


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


def _fp32_gqa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    q = query.float().transpose(1, 2)
    group_size = query.shape[2] // key.shape[2]
    k = key.float().transpose(1, 2).repeat_interleave(group_size, dim=1)
    v = value.float().transpose(1, 2).repeat_interleave(group_size, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if causal:
        causal_mask = torch.ones(
            query.shape[1], key.shape[1], device=query.device, dtype=torch.bool
        ).tril()
        scores = scores.masked_fill(~causal_mask, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
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
    *,
    causal: bool = True,
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
                is_causal=causal,
                dropout_p=0.0,
                enable_gqa=True,
            ).transpose(1, 2)

    return run


def _graph_capture(
    function: Callable[[], torch.Tensor],
) -> tuple[Callable[[], torch.Tensor], torch.Tensor, int]:
    # Autograd capture requires warmup on a side stream so allocations and
    # library handles are initialized outside the capture stream's history.
    current = torch.cuda.current_stream()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(current)
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            function()
    current.wait_stream(warmup_stream)
    current.synchronize()
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


def _record_contiguous_baseline(
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
    atol: float = 0.03,
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
                correctness_passed=eager_errors[0] <= atol,
                failure_reason=(
                    None if eager_errors[0] <= atol else "tolerance_exceeded"
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
                correctness_passed=graph_errors[0] <= atol,
                failure_reason=(
                    None if graph_errors[0] <= atol else "tolerance_exceeded"
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


def _profile_sm80_prefill(
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[BackendEvidence, ...]:
    cell_id = "sm80_prefill_b1_2k_g4_d128"
    environment = cuda_environment_fingerprint(
        library_names=("flash-attn", "flashinfer-python", "flashinfer-cubin")
    )
    query = torch.randn(1, 2048, 32, 128, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 2048, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    reference = _fp32_gqa_reference(query, key, value)
    rows: list[BackendEvidence] = []
    for tile_q, tile_k in ((64, 32), (64, 64), (128, 32), (128, 64), (128, 128)):
        config = f"q{tile_q}_k{tile_k}"
        module = FusedOnlineAttention(
            num_heads=32,
            num_kv_heads=8,
            head_dim=128,
            tile_size_q=tile_q,
            tile_size_k=tile_k,
            device=query.device,
            dtype=query.dtype,
        ).eval()

        def head_private_fn(*, _module=module) -> torch.Tensor:
            return _module(query, key, value, causal=True)

        try:
            with torch.no_grad():
                eager_output = head_private_fn()
                torch.cuda.synchronize()
                backend = str(module.last_backend_used)
                if not backend.startswith("triton"):
                    raise RuntimeError(f"resolved to non-native backend {backend}")
                eager_errors = _errors(eager_output, reference)
                eager_samples = _time_average_samples(
                    head_private_fn,
                    warmup=warmup,
                    iterations=iterations,
                    repeats=repeats,
                )
                rows.append(
                    measured_evidence(
                        evidence_id=(
                            f"{cell_id}:streamattn:online-softmax:{config}:eager"
                        ),
                        cell_id=cell_id,
                        provider="streamattn",
                        requested_backend="triton_online_softmax_exact",
                        resolved_backend=f"{backend}:eager",
                        environment=environment,
                        samples_ms=eager_samples,
                        correctness_reference="fp32_dense_causal_gqa",
                        checked_cases=reference.numel(),
                        max_abs_error=eager_errors[0],
                        max_relative_error=eager_errors[1],
                        workspace_bytes=0,
                        supported_range=_supported_range(cell_id),
                        native=True,
                        correctness_passed=eager_errors[0] <= 0.015,
                        failure_reason=(
                            None
                            if eager_errors[0] <= 0.015
                            else "fp16_tolerance_exceeded"
                        ),
                        family_id="triton_online_softmax_exact",
                        kernel_key=config,
                        timed_allocation_count=1,
                        detail="eager module allocates its output",
                    )
                )
                replay, graph_output, graph_workspace = _graph_capture(
                    head_private_fn
                )
                graph_errors = _errors(graph_output, reference)
                graph_samples = _time_average_samples(
                    replay,
                    warmup=warmup,
                    iterations=iterations,
                    repeats=repeats,
                )
                rows.append(
                    measured_evidence(
                        evidence_id=(
                            f"{cell_id}:streamattn:online-softmax:{config}:cuda-graph"
                        ),
                        cell_id=cell_id,
                        provider="streamattn",
                        requested_backend="triton_online_softmax_exact",
                        resolved_backend=f"{backend}:cuda_graph_replay",
                        environment=environment,
                        samples_ms=graph_samples,
                        correctness_reference="fp32_dense_causal_gqa",
                        checked_cases=reference.numel(),
                        max_abs_error=graph_errors[0],
                        max_relative_error=graph_errors[1],
                        workspace_bytes=graph_workspace,
                        supported_range=_supported_range(cell_id),
                        native=True,
                        correctness_passed=graph_errors[0] <= 0.015,
                        failure_reason=(
                            None
                            if graph_errors[0] <= 0.015
                            else "fp16_tolerance_exceeded"
                        ),
                        family_id="triton_online_softmax_exact",
                        kernel_key=f"{config}:cuda_graph",
                        detail="fixed-address CUDA graph replay",
                    )
                )
        except Exception as exc:
            torch.cuda.synchronize()
            rows.append(
                outcome_evidence(
                    evidence_id=(
                        f"{cell_id}:streamattn:online-softmax:{config}:error"
                    ),
                    cell_id=cell_id,
                    provider="streamattn",
                    requested_backend="triton_online_softmax_exact",
                    environment=environment,
                    status=MeasurementStatus.ERROR,
                    detail=f"{type(exc).__name__}: {exc}",
                    native=True,
                    family_id="triton_online_softmax_exact",
                )
            )
    configurations = (
        (1, 64, 64, 4, 2),
        (1, 64, 128, 4, 3),
        (2, 32, 64, 4, 2),
        (2, 32, 128, 8, 3),
        (4, 16, 64, 4, 2),
        (4, 16, 128, 8, 3),
        (4, 32, 64, 8, 2),
        (4, 32, 128, 8, 3),
    )
    for heads_per_program, tile_m, tile_n, num_warps, num_stages in configurations:
        config = (
            f"h{heads_per_program}_m{tile_m}_n{tile_n}_"
            f"w{num_warps}_s{num_stages}"
        )
        output = torch.empty_like(query)
        lse = torch.empty((1, 32, 2048), device="cuda", dtype=torch.float32)

        def native_fn(
            *,
            _heads=heads_per_program,
            _tile_m=tile_m,
            _tile_n=tile_n,
            _warps=num_warps,
            _stages=num_stages,
            _output=output,
            _lse=lse,
        ) -> torch.Tensor:
            return grouped_gqa_prefill(
                query,
                key,
                value,
                heads_per_program=_heads,
                tile_m=_tile_m,
                tile_n=_tile_n,
                num_warps=_warps,
                num_stages=_stages,
                output=_output,
                lse=_lse,
            )

        try:
            native_output = native_fn()
            torch.cuda.synchronize()
            errors = _errors(native_output, reference)
            direct_samples = _time_average_samples(
                native_fn,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )
            workspace_bytes = (
                output.numel() * output.element_size()
                + lse.numel() * lse.element_size()
            )
            for suffix, samples, resolved, detail in (
                (
                    "direct",
                    direct_samples,
                    "triton_grouped_gqa_prefill",
                    "preallocated direct Triton launch",
                ),
            ):
                rows.append(
                    measured_evidence(
                        evidence_id=f"{cell_id}:streamattn:{config}:{suffix}",
                        cell_id=cell_id,
                        provider="streamattn",
                        requested_backend="triton_grouped_gqa_prefill",
                        resolved_backend=resolved,
                        environment=environment,
                        samples_ms=samples,
                        correctness_reference="fp32_dense_causal_gqa",
                        checked_cases=reference.numel(),
                        max_abs_error=errors[0],
                        max_relative_error=errors[1],
                        workspace_bytes=workspace_bytes,
                        supported_range=_supported_range(cell_id),
                        native=True,
                        correctness_passed=errors[0] <= 0.015,
                        failure_reason=(
                            None if errors[0] <= 0.015 else "fp16_tolerance_exceeded"
                        ),
                        family_id="triton_grouped_gqa_prefill",
                        kernel_key=config,
                        detail=detail,
                    )
                )
            replay, graph_output, graph_workspace = _graph_capture(native_fn)
            graph_errors = _errors(graph_output, reference)
            graph_samples = _time_average_samples(
                replay,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )
            rows.append(
                measured_evidence(
                    evidence_id=f"{cell_id}:streamattn:{config}:cuda-graph",
                    cell_id=cell_id,
                    provider="streamattn",
                    requested_backend="triton_grouped_gqa_prefill",
                    resolved_backend="triton_grouped_gqa_prefill:cuda_graph_replay",
                    environment=environment,
                    samples_ms=graph_samples,
                    correctness_reference="fp32_dense_causal_gqa",
                    checked_cases=reference.numel(),
                    max_abs_error=graph_errors[0],
                    max_relative_error=graph_errors[1],
                    workspace_bytes=workspace_bytes + graph_workspace,
                    supported_range=_supported_range(cell_id),
                    native=True,
                    correctness_passed=graph_errors[0] <= 0.015,
                    failure_reason=(
                        None
                        if graph_errors[0] <= 0.015
                        else "fp16_tolerance_exceeded"
                    ),
                    family_id="triton_grouped_gqa_prefill",
                    kernel_key=f"{config}:cuda_graph",
                    detail="fixed-address CUDA graph replay",
                )
            )
        except Exception as exc:
            torch.cuda.synchronize()
            rows.append(
                outcome_evidence(
                    evidence_id=f"{cell_id}:streamattn:{config}:error",
                    cell_id=cell_id,
                    provider="streamattn",
                    requested_backend="triton_grouped_gqa_prefill",
                    environment=environment,
                    status=MeasurementStatus.ERROR,
                    detail=f"{type(exc).__name__}: {exc}",
                    native=True,
                    family_id="triton_grouped_gqa_prefill",
                )
            )

    flash_backend = None if SDPBackend is None else SDPBackend.FLASH_ATTENTION
    cudnn_backend = (
        None if SDPBackend is None else getattr(SDPBackend, "CUDNN_ATTENTION", None)
    )
    for backend_name, resolved_backend, backend in (
        ("pytorch_sdpa", "torch_sdpa_flash_attention", flash_backend),
        ("cudnn_sdpa", "torch_sdpa_cudnn_attention", cudnn_backend),
    ):
        if backend is None:
            rows.append(
                outcome_evidence(
                    evidence_id=f"{cell_id}:external:{backend_name}:unsupported",
                    cell_id=cell_id,
                    provider="external",
                    requested_backend=backend_name,
                    environment=environment,
                    status=MeasurementStatus.UNSUPPORTED,
                    detail=f"PyTorch build does not expose {backend_name}",
                )
            )
        else:
            _record_contiguous_baseline(
                rows,
                cell_id=cell_id,
                backend_name=backend_name,
                resolved_backend=resolved_backend,
                function=_sdpa_function(query, key, value, backend),
                reference=reference,
                environment=environment,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
                atol=0.015,
            )
    try:
        from flash_attn import flash_attn_func

        def flashattention_fn() -> torch.Tensor:
            return flash_attn_func(query, key, value, causal=True)

        _record_contiguous_baseline(
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
            atol=0.015,
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
    try:
        import flashinfer

        def flashinfer_fn() -> torch.Tensor:
            return flashinfer.single_prefill_with_kv_cache(
                query[0], key[0], value[0], causal=True, kv_layout="NHD"
            ).unsqueeze(0)

        _record_contiguous_baseline(
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
            atol=0.015,
        )
    except Exception as exc:
        rows.append(
            outcome_evidence(
                evidence_id=f"{cell_id}:external:flashinfer_prefill:unsupported",
                cell_id=cell_id,
                provider="external",
                requested_backend="flashinfer_prefill",
                environment=environment,
                status=MeasurementStatus.UNSUPPORTED,
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
    return tuple(rows)


def _profile_sm80_tail_prefill(
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[BackendEvidence, ...]:
    spec = SM80_ADDITIONAL_PREFILL_CELLS[0]
    cell_id = str(spec["cell_id"])
    environment = cuda_environment_fingerprint(
        library_names=("flash-attn", "flashinfer-python", "flashinfer-cubin")
    )
    batch = int(spec["batch"])
    seq_len = int(spec["seq_len"])
    q_heads = int(spec["q_heads"])
    kv_heads = int(spec["kv_heads"])
    head_dim = int(spec["head_dim"])
    dtype = spec["dtype"]
    query = torch.randn(batch, seq_len, q_heads, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(batch, seq_len, kv_heads, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    reference = _fp32_gqa_reference(query, key, value)
    rows: list[BackendEvidence] = []
    configurations = (
        (1, 64, 64, 4, 2),
        (2, 32, 64, 4, 2),
        (4, 16, 64, 4, 2),
        (4, 32, 64, 8, 2),
    )
    for heads_per_program, tile_m, tile_n, num_warps, num_stages in configurations:
        config = (
            f"h{heads_per_program}_m{tile_m}_n{tile_n}_"
            f"w{num_warps}_s{num_stages}"
        )
        output = torch.empty_like(query)
        lse = torch.empty((batch, q_heads, seq_len), device="cuda", dtype=torch.float32)

        def native_fn(
            *,
            _heads=heads_per_program,
            _tile_m=tile_m,
            _tile_n=tile_n,
            _warps=num_warps,
            _stages=num_stages,
            _output=output,
            _lse=lse,
        ) -> torch.Tensor:
            return grouped_gqa_prefill(
                query,
                key,
                value,
                heads_per_program=_heads,
                tile_m=_tile_m,
                tile_n=_tile_n,
                num_warps=_warps,
                num_stages=_stages,
                output=_output,
                lse=_lse,
            )

        try:
            native_output = native_fn()
            torch.cuda.synchronize()
            errors = _errors(native_output, reference)
            replay, graph_output, graph_workspace = _graph_capture(native_fn)
            graph_errors = _errors(graph_output, reference)
            samples = _time_average_samples(
                replay,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )
            rows.append(
                measured_evidence(
                    evidence_id=f"{cell_id}:streamattn:{config}:cuda-graph",
                    cell_id=cell_id,
                    provider="streamattn",
                    requested_backend="triton_grouped_gqa_prefill",
                    resolved_backend="triton_grouped_gqa_prefill:cuda_graph_replay",
                    environment=environment,
                    samples_ms=samples,
                    correctness_reference="fp32_dense_causal_gqa",
                    checked_cases=reference.numel(),
                    max_abs_error=max(errors[0], graph_errors[0]),
                    max_relative_error=max(errors[1], graph_errors[1]),
                    workspace_bytes=(
                        output.numel() * output.element_size()
                        + lse.numel() * lse.element_size()
                        + graph_workspace
                    ),
                    supported_range=_supported_range(cell_id),
                    native=True,
                    correctness_passed=max(errors[0], graph_errors[0]) <= 0.03,
                    failure_reason=(
                        None
                        if max(errors[0], graph_errors[0]) <= 0.03
                        else "bf16_tolerance_exceeded"
                    ),
                    family_id="triton_grouped_gqa_prefill",
                    kernel_key=f"{config}:cuda_graph",
                    detail="fixed-address CUDA graph replay",
                )
            )
        except Exception as exc:
            torch.cuda.synchronize()
            rows.append(
                outcome_evidence(
                    evidence_id=f"{cell_id}:streamattn:{config}:error",
                    cell_id=cell_id,
                    provider="streamattn",
                    requested_backend="triton_grouped_gqa_prefill",
                    environment=environment,
                    status=MeasurementStatus.ERROR,
                    detail=f"{type(exc).__name__}: {exc}",
                    native=True,
                    family_id="triton_grouped_gqa_prefill",
                )
            )
    flash_backend = None if SDPBackend is None else SDPBackend.FLASH_ATTENTION
    cudnn_backend = (
        None if SDPBackend is None else getattr(SDPBackend, "CUDNN_ATTENTION", None)
    )
    for backend_name, resolved_backend, backend in (
        ("pytorch_sdpa", "torch_sdpa_flash_attention", flash_backend),
        ("cudnn_sdpa", "torch_sdpa_cudnn_attention", cudnn_backend),
    ):
        if backend is None:
            rows.append(
                outcome_evidence(
                    evidence_id=f"{cell_id}:external:{backend_name}:unsupported",
                    cell_id=cell_id,
                    provider="external",
                    requested_backend=backend_name,
                    environment=environment,
                    status=MeasurementStatus.UNSUPPORTED,
                    detail=f"PyTorch build does not expose {backend_name}",
                )
            )
        else:
            _record_contiguous_baseline(
                rows,
                cell_id=cell_id,
                backend_name=backend_name,
                resolved_backend=resolved_backend,
                function=_sdpa_function(query, key, value, backend),
                reference=reference,
                environment=environment,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )
    try:
        from flash_attn import flash_attn_func

        _record_contiguous_baseline(
            rows,
            cell_id=cell_id,
            backend_name="flashattention",
            resolved_backend="flash_attn_func",
            function=lambda: flash_attn_func(query, key, value, causal=True),
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
    try:
        import flashinfer

        _record_contiguous_baseline(
            rows,
            cell_id=cell_id,
            backend_name="flashinfer_prefill",
            resolved_backend="flashinfer_single_prefill",
            function=lambda: flashinfer.single_prefill_with_kv_cache(
                query[0], key[0], value[0], causal=True, kv_layout="NHD"
            ).unsqueeze(0),
            reference=reference,
            environment=environment,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
    except Exception as exc:
        rows.append(
            outcome_evidence(
                evidence_id=f"{cell_id}:external:flashinfer_prefill:unsupported",
                cell_id=cell_id,
                provider="external",
                requested_backend="flashinfer_prefill",
                environment=environment,
                status=MeasurementStatus.UNSUPPORTED,
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
    return tuple(rows)


def _profile_sm80_hnd_noncausal_prefill(
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[BackendEvidence, ...]:
    spec = SM80_ADDITIONAL_PREFILL_CELLS[1]
    cell_id = str(spec["cell_id"])
    environment = cuda_environment_fingerprint(
        library_names=("flash-attn", "flashinfer-python", "flashinfer-cubin")
    )
    batch = int(spec["batch"])
    seq_len = int(spec["seq_len"])
    heads = int(spec["q_heads"])
    head_dim = int(spec["head_dim"])
    dtype = spec["dtype"]
    query_hnd = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=dtype)
    key_hnd = torch.randn_like(query_hnd)
    value_hnd = torch.randn_like(query_hnd)
    query = query_hnd.transpose(1, 2)
    key = key_hnd.transpose(1, 2)
    value = value_hnd.transpose(1, 2)
    reference = _fp32_gqa_reference(query, key, value, causal=False)
    rows: list[BackendEvidence] = []
    for tile_q, tile_k in ((64, 32), (64, 64), (128, 32), (128, 64)):
        config = f"q{tile_q}_k{tile_k}:hnd-strided"
        module = FusedOnlineAttention(
            num_heads=heads,
            num_kv_heads=heads,
            head_dim=head_dim,
            tile_size_q=tile_q,
            tile_size_k=tile_k,
            device=query.device,
            dtype=query.dtype,
        ).eval()

        def native_fn(*, _module=module) -> torch.Tensor:
            return _module(query, key, value, causal=False)

        try:
            with torch.no_grad():
                native_fn()
                torch.cuda.synchronize()
                if not str(module.last_backend_used).startswith("triton"):
                    raise RuntimeError(
                        f"resolved to non-native backend {module.last_backend_used}"
                    )
                replay, graph_output, graph_workspace = _graph_capture(native_fn)
                errors = _errors(graph_output, reference)
                samples = _time_average_samples(
                    replay,
                    warmup=warmup,
                    iterations=iterations,
                    repeats=repeats,
                )
            rows.append(
                measured_evidence(
                    evidence_id=f"{cell_id}:streamattn:{config}:cuda-graph",
                    cell_id=cell_id,
                    provider="streamattn",
                    requested_backend="triton_online_softmax_exact",
                    resolved_backend=(
                        f"{module.last_backend_used}:hnd_strided:cuda_graph_replay"
                    ),
                    environment=environment,
                    samples_ms=samples,
                    correctness_reference="fp32_dense_noncausal_mha",
                    checked_cases=reference.numel(),
                    max_abs_error=errors[0],
                    max_relative_error=errors[1],
                    workspace_bytes=graph_workspace,
                    supported_range=_supported_range(cell_id),
                    native=True,
                    correctness_passed=errors[0] <= 0.015,
                    failure_reason=(
                        None if errors[0] <= 0.015 else "fp16_tolerance_exceeded"
                    ),
                    family_id="triton_online_softmax_exact",
                    kernel_key=f"{config}:cuda_graph",
                    detail=(
                        "fixed-address CUDA graph; any HND-to-kernel layout work is "
                        "inside the timed native call"
                    ),
                )
            )
        except Exception as exc:
            torch.cuda.synchronize()
            rows.append(
                outcome_evidence(
                    evidence_id=f"{cell_id}:streamattn:{config}:error",
                    cell_id=cell_id,
                    provider="streamattn",
                    requested_backend="triton_online_softmax_exact",
                    environment=environment,
                    status=MeasurementStatus.ERROR,
                    detail=f"{type(exc).__name__}: {exc}",
                    native=True,
                    family_id="triton_online_softmax_exact",
                )
            )

    def hnd_sdpa(backend) -> Callable[[], torch.Tensor]:
        def run() -> torch.Tensor:
            context = (
                contextlib.nullcontext()
                if sdpa_kernel is None or backend is None
                else sdpa_kernel(backend)
            )
            with context:
                return F.scaled_dot_product_attention(
                    query_hnd,
                    key_hnd,
                    value_hnd,
                    is_causal=False,
                    dropout_p=0.0,
                ).transpose(1, 2)

        return run

    flash_backend = None if SDPBackend is None else SDPBackend.FLASH_ATTENTION
    cudnn_backend = (
        None if SDPBackend is None else getattr(SDPBackend, "CUDNN_ATTENTION", None)
    )
    for backend_name, resolved_backend, backend in (
        ("pytorch_sdpa", "torch_sdpa_flash_attention", flash_backend),
        ("cudnn_sdpa", "torch_sdpa_cudnn_attention", cudnn_backend),
    ):
        if backend is None:
            rows.append(
                outcome_evidence(
                    evidence_id=f"{cell_id}:external:{backend_name}:unsupported",
                    cell_id=cell_id,
                    provider="external",
                    requested_backend=backend_name,
                    environment=environment,
                    status=MeasurementStatus.UNSUPPORTED,
                    detail=f"PyTorch build does not expose {backend_name}",
                )
            )
        else:
            _record_contiguous_baseline(
                rows,
                cell_id=cell_id,
                backend_name=backend_name,
                resolved_backend=resolved_backend,
                function=hnd_sdpa(backend),
                reference=reference,
                environment=environment,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
                atol=0.015,
            )
    for backend_name, detail in (
        ("flashattention", "flash_attn_func requires BSHD and has no direct HND contract"),
        ("flashinfer_prefill", "single-prefill has no direct batched HND query contract"),
    ):
        rows.append(
            outcome_evidence(
                evidence_id=f"{cell_id}:external:{backend_name}:direct-hnd-unsupported",
                cell_id=cell_id,
                provider="external",
                requested_backend=backend_name,
                environment=environment,
                status=MeasurementStatus.UNSUPPORTED,
                detail=detail,
            )
        )
    return tuple(rows)


def _training_errors(
    result: tuple[torch.Tensor, ...],
    reference: tuple[torch.Tensor, ...],
) -> tuple[float, float]:
    errors = [_errors(actual, expected) for actual, expected in zip(result, reference)]
    return max(error[0] for error in errors), max(error[1] for error in errors)


def _sdpa_training_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
    backend,
    *,
    dropout_p: float,
) -> Callable[[], tuple[torch.Tensor, ...]]:
    def run() -> tuple[torch.Tensor, ...]:
        context = (
            contextlib.nullcontext()
            if sdpa_kernel is None or backend is None
            else sdpa_kernel(backend)
        )
        with torch.enable_grad(), context:
            output_hnd = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                is_causal=True,
                dropout_p=dropout_p,
            )
            output = output_hnd.transpose(1, 2)
            gradients = torch.autograd.grad(
                output,
                (query, key, value),
                grad_output,
                allow_unused=False,
            )
        return (output, *gradients)

    return run


def _sdpa_training_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    backend,
    *,
    dropout_p: float,
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        context = (
            contextlib.nullcontext()
            if sdpa_kernel is None or backend is None
            else sdpa_kernel(backend)
        )
        with torch.enable_grad(), context:
            return F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                is_causal=True,
                dropout_p=dropout_p,
            ).transpose(1, 2)

    return run


def _fp32_training_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    q = query.detach().float().requires_grad_(True)
    k = key.detach().float().requires_grad_(True)
    v = value.detach().float().requires_grad_(True)
    with torch.enable_grad():
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            dropout_p=0.0,
        ).transpose(1, 2)
        gradients = torch.autograd.grad(
            output,
            (q, k, v),
            grad_output.float(),
            allow_unused=False,
        )
    return (output.detach(), *(gradient.detach() for gradient in gradients))


def _capture_training_backward(
    forward: Callable[[], torch.Tensor],
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    grad_output: torch.Tensor,
) -> tuple[
    Callable[[], tuple[torch.Tensor, ...]],
    tuple[torch.Tensor, ...],
    int,
]:
    current = torch.cuda.current_stream()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(current)
    with torch.cuda.stream(warmup_stream), torch.enable_grad():
        for _ in range(3):
            for tensor in inputs:
                tensor.grad = None
            forward().backward(grad_output)
    current.wait_stream(warmup_stream)
    current.synchronize()
    for tensor in inputs:
        tensor.grad = torch.zeros_like(tensor)
    before = torch.cuda.memory_allocated()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.enable_grad():
        for tensor in inputs:
            tensor.grad.zero_()
        output = forward()
        output.backward(grad_output)
    torch.cuda.synchronize()
    workspace = max(0, torch.cuda.memory_allocated() - before)

    def result() -> tuple[torch.Tensor, ...]:
        return (output, *(tensor.grad for tensor in inputs))

    def replay() -> tuple[torch.Tensor, ...]:
        graph.replay()
        return result()

    replay()
    torch.cuda.synchronize()
    return replay, result(), workspace


def _record_training_backend(
    rows: list[BackendEvidence],
    *,
    cell_id: str,
    provider: str,
    backend_name: str,
    resolved_backend: str,
    function: Callable[[], tuple[torch.Tensor, ...]],
    reference: tuple[torch.Tensor, ...] | None,
    environment,
    warmup: int,
    iterations: int,
    repeats: int,
    tolerance: float,
    native: bool,
    family_id: str | None = None,
    kernel_key: str | None = None,
    deterministic_self_check: bool = False,
    graph_forward: Callable[[], torch.Tensor] | None = None,
    graph_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    graph_grad_output: torch.Tensor | None = None,
) -> None:
    try:
        if deterministic_self_check:
            torch.manual_seed(739)
            first = function()
            torch.cuda.synchronize()
            torch.manual_seed(739)
            second = function()
            torch.cuda.synchronize()
            errors = _training_errors(first, second)
            correctness_reference = "same-seed deterministic backend replay"
        else:
            if reference is None:
                raise ValueError("training reference is required")
            first = function()
            torch.cuda.synchronize()
            errors = _training_errors(first, reference)
            correctness_reference = "fp32_sdpa_forward_dq_dk_dv"
        eager_samples = _time_average_samples(
            function,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        rows.append(
            measured_evidence(
                evidence_id=f"{cell_id}:{provider}:{backend_name}:eager-allocating",
                cell_id=cell_id,
                provider=provider,
                requested_backend=backend_name,
                resolved_backend=f"{resolved_backend}:eager",
                environment=environment,
                samples_ms=eager_samples,
                correctness_reference=correctness_reference,
                checked_cases=sum(tensor.numel() for tensor in first),
                max_abs_error=errors[0],
                max_relative_error=errors[1],
                workspace_bytes=0,
                supported_range=_supported_range(cell_id),
                native=native,
                correctness_passed=errors[0] <= tolerance,
                failure_reason=(None if errors[0] <= tolerance else "tolerance_exceeded"),
                family_id=family_id,
                kernel_key=kernel_key,
                timed_allocation_count=1,
                detail="eager forward and dQ/dK/dV allocate autograd outputs",
            )
        )
        if (
            graph_forward is not None
            and graph_inputs is not None
            and graph_grad_output is not None
        ):
            replay, graph_result, graph_workspace = _capture_training_backward(
                graph_forward,
                graph_inputs,
                graph_grad_output,
            )
        else:
            replay, graph_result, graph_workspace = _graph_capture(function)
        if deterministic_self_check:
            graph_errors = errors
        else:
            assert reference is not None
            graph_errors = _training_errors(graph_result, reference)
        graph_samples = _time_average_samples(
            replay,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        rows.append(
            measured_evidence(
                evidence_id=f"{cell_id}:{provider}:{backend_name}:cuda-graph",
                cell_id=cell_id,
                provider=provider,
                requested_backend=backend_name,
                resolved_backend=f"{resolved_backend}:cuda_graph_replay",
                environment=environment,
                samples_ms=graph_samples,
                correctness_reference=correctness_reference,
                checked_cases=sum(tensor.numel() for tensor in graph_result),
                max_abs_error=graph_errors[0],
                max_relative_error=graph_errors[1],
                workspace_bytes=graph_workspace,
                supported_range=_supported_range(cell_id),
                native=native,
                correctness_passed=graph_errors[0] <= tolerance,
                failure_reason=(
                    None if graph_errors[0] <= tolerance else "tolerance_exceeded"
                ),
                family_id=family_id,
                kernel_key=(None if kernel_key is None else f"{kernel_key}:cuda_graph"),
                detail="fixed-address CUDA graph replays forward and dQ/dK/dV",
            )
        )
    except Exception as exc:
        torch.cuda.synchronize()
        rows.append(
            outcome_evidence(
                evidence_id=f"{cell_id}:{provider}:{backend_name}:unsupported",
                cell_id=cell_id,
                provider=provider,
                requested_backend=backend_name,
                environment=environment,
                status=MeasurementStatus.UNSUPPORTED,
                detail=f"{type(exc).__name__}: {exc}",
                native=native,
                family_id=family_id,
            )
        )


def _profile_sm80_training(
    *,
    warmup: int,
    iterations: int,
    repeats: int,
    skip_native: bool = False,
) -> tuple[BackendEvidence, ...]:
    environment = cuda_environment_fingerprint(
        library_names=("flash-attn", "flashinfer-python", "flashinfer-cubin")
    )
    rows: list[BackendEvidence] = []
    cell_id = "sm80_train_b1_2k_mha_d128"
    query = torch.randn(
        1, 2048, 32, 128, device="cuda", dtype=torch.float16, requires_grad=True
    )
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    grad_output = torch.randn_like(query)
    reference = _fp32_training_reference(query, key, value, grad_output)
    native_tiles = () if skip_native else ((64, 32), (64, 64), (128, 32), (128, 64))
    for tile_q, tile_k in native_tiles:
        config = f"q{tile_q}_k{tile_k}"
        module = FusedOnlineAttention(
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            tile_size_q=tile_q,
            tile_size_k=tile_k,
            device=query.device,
            dtype=query.dtype,
        ).train()

        def native_forward(*, _module=module) -> torch.Tensor:
            return _module(query, key, value, causal=True)

        def native_step(*, _module=module) -> tuple[torch.Tensor, ...]:
            with torch.enable_grad():
                output = native_forward(_module=_module)
                gradients = torch.autograd.grad(
                    output,
                    (query, key, value),
                    grad_output,
                    allow_unused=False,
                )
            return (output, *gradients)

        _record_training_backend(
            rows,
            cell_id=cell_id,
            provider="streamattn",
            backend_name=f"online-softmax-{config}",
            resolved_backend="triton_online_softmax_autograd",
            function=native_step,
            reference=reference,
            environment=environment,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
            tolerance=0.03,
            native=True,
            family_id="triton_online_softmax_exact_train",
            kernel_key=config,
            graph_forward=native_forward,
            graph_inputs=(query, key, value),
            graph_grad_output=grad_output,
        )
    flash_backend = None if SDPBackend is None else SDPBackend.FLASH_ATTENTION
    cudnn_backend = (
        None if SDPBackend is None else getattr(SDPBackend, "CUDNN_ATTENTION", None)
    )
    for backend_name, resolved_backend, backend in (
        ("pytorch_sdpa", "torch_sdpa_flash_attention", flash_backend),
        ("cudnn_sdpa", "torch_sdpa_cudnn_attention", cudnn_backend),
    ):
        if backend is None:
            rows.append(
                outcome_evidence(
                    evidence_id=f"{cell_id}:external:{backend_name}:unsupported",
                    cell_id=cell_id,
                    provider="external",
                    requested_backend=backend_name,
                    environment=environment,
                    status=MeasurementStatus.UNSUPPORTED,
                    detail=f"PyTorch build does not expose {backend_name}",
                )
            )
        else:
            baseline_forward = _sdpa_training_forward(
                query,
                key,
                value,
                backend,
                dropout_p=0.0,
            )
            _record_training_backend(
                rows,
                cell_id=cell_id,
                provider="external",
                backend_name=backend_name,
                resolved_backend=resolved_backend,
                function=_sdpa_training_step(
                    query,
                    key,
                    value,
                    grad_output,
                    backend,
                    dropout_p=0.0,
                ),
                reference=reference,
                environment=environment,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
                tolerance=0.03,
                native=False,
                graph_forward=baseline_forward,
                graph_inputs=(query, key, value),
                graph_grad_output=grad_output,
            )
    try:
        from flash_attn import flash_attn_func

        def flashattention_step() -> tuple[torch.Tensor, ...]:
            with torch.enable_grad():
                output = flash_attn_func(query, key, value, causal=True)
                gradients = torch.autograd.grad(
                    output,
                    (query, key, value),
                    grad_output,
                    allow_unused=False,
                )
            return (output, *gradients)

        def flashattention_forward() -> torch.Tensor:
            return flash_attn_func(query, key, value, causal=True)

        _record_training_backend(
            rows,
            cell_id=cell_id,
            provider="external",
            backend_name="flashattention",
            resolved_backend="flash_attn_func",
            function=flashattention_step,
            reference=reference,
            environment=environment,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
            tolerance=0.03,
            native=False,
            graph_forward=flashattention_forward,
            graph_inputs=(query, key, value),
            graph_grad_output=grad_output,
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

    return tuple(rows)


def _profile_sm80_training_dropout(
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> tuple[BackendEvidence, ...]:
    environment = cuda_environment_fingerprint(
        library_names=("flash-attn", "flashinfer-python", "flashinfer-cubin")
    )
    rows: list[BackendEvidence] = []
    dropout_cell = "sm80_train_dropout_mha_d128"
    rows.append(
        outcome_evidence(
            evidence_id=f"{dropout_cell}:streamattn:native-dropout-unsupported",
            cell_id=dropout_cell,
            provider="streamattn",
            requested_backend="triton_online_softmax_exact_train",
            environment=environment,
            status=MeasurementStatus.UNSUPPORTED,
            detail=(
                "native backward intentionally excludes dropout until the forward "
                "Philox mask is replayed exactly in dQ/dK/dV"
            ),
            native=True,
            family_id="triton_online_softmax_exact_train",
        )
    )
    dropout_query = torch.randn(
        1, 1024, 16, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    dropout_key = torch.randn_like(dropout_query, requires_grad=True)
    dropout_value = torch.randn_like(dropout_query, requires_grad=True)
    dropout_grad = torch.randn_like(dropout_query)
    math_backend = None if SDPBackend is None else SDPBackend.MATH
    for backend_name, resolved_backend, backend in (
        ("pytorch_sdpa", "torch_sdpa_math", math_backend),
    ):
        if backend is None:
            rows.append(
                outcome_evidence(
                    evidence_id=f"{dropout_cell}:external:{backend_name}:unsupported",
                    cell_id=dropout_cell,
                    provider="external",
                    requested_backend=backend_name,
                    environment=environment,
                    status=MeasurementStatus.UNSUPPORTED,
                    detail=f"PyTorch build does not expose {backend_name}",
                )
            )
        else:
            baseline_forward = _sdpa_training_forward(
                dropout_query,
                dropout_key,
                dropout_value,
                backend,
                dropout_p=0.1,
            )
            _record_training_backend(
                rows,
                cell_id=dropout_cell,
                provider="external",
                backend_name=backend_name,
                resolved_backend=resolved_backend,
                function=_sdpa_training_step(
                    dropout_query,
                    dropout_key,
                    dropout_value,
                    dropout_grad,
                    backend,
                    dropout_p=0.1,
                ),
                reference=None,
                environment=environment,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
                tolerance=0.0,
                native=False,
                deterministic_self_check=True,
                graph_forward=baseline_forward,
                graph_inputs=(dropout_query, dropout_key, dropout_value),
                graph_grad_output=dropout_grad,
            )
    rows.append(
        outcome_evidence(
            evidence_id=f"{dropout_cell}:external:cudnn_sdpa:unsupported",
            cell_id=dropout_cell,
            provider="external",
            requested_backend="cudnn_sdpa",
            environment=environment,
            status=MeasurementStatus.UNSUPPORTED,
            detail=(
                "deterministic dropout backward uses the PyTorch math baseline; "
                "cuDNN RNG replay is backend-specific and is not directly comparable"
            ),
        )
    )
    try:
        from flash_attn import flash_attn_func

        def flashattention_dropout_step() -> tuple[torch.Tensor, ...]:
            with torch.enable_grad():
                output = flash_attn_func(
                    dropout_query,
                    dropout_key,
                    dropout_value,
                    dropout_p=0.1,
                    causal=True,
                )
                gradients = torch.autograd.grad(
                    output,
                    (dropout_query, dropout_key, dropout_value),
                    dropout_grad,
                    allow_unused=False,
                )
            return (output, *gradients)

        def flashattention_dropout_forward() -> torch.Tensor:
            return flash_attn_func(
                dropout_query,
                dropout_key,
                dropout_value,
                dropout_p=0.1,
                causal=True,
            )

        _record_training_backend(
            rows,
            cell_id=dropout_cell,
            provider="external",
            backend_name="flashattention",
            resolved_backend="flash_attn_func",
            function=flashattention_dropout_step,
            reference=None,
            environment=environment,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
            tolerance=0.0,
            native=False,
            deterministic_self_check=True,
            graph_forward=flashattention_dropout_forward,
            graph_inputs=(dropout_query, dropout_key, dropout_value),
            graph_grad_output=dropout_grad,
        )
    except Exception as exc:
        rows.append(
            outcome_evidence(
                evidence_id=f"{dropout_cell}:external:flashattention:unsupported",
                cell_id=dropout_cell,
                provider="external",
                requested_backend="flashattention",
                environment=environment,
                status=MeasurementStatus.UNSUPPORTED,
                detail=f"{type(exc).__name__}: {exc}",
            )
        )
    return tuple(rows)


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
        reference = _fp32_gqa_reference(query, key, value)
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
            _record_contiguous_baseline(
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
            _record_contiguous_baseline(
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
            _record_contiguous_baseline(
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
                _record_contiguous_baseline(
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
    parser.add_argument(
        "--architecture", choices=("sm80", "sm90", "sm100"), required=True
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--build-dir", type=Path)
    args = parser.parse_args()
    if args.architecture == "sm80":
        evidence = profile_sm80(warmup=args.warmup, repeats=args.repeats)
    elif args.architecture == "sm90":
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
