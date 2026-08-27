"""Profile native StreamAttn GQA prefill/training against exact SDPA backends."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from stream_attention.core.fused_online_attention import FusedOnlineAttention

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older PyTorch
    SDPBackend = None
    sdpa_kernel = None


def _elapsed_ms(fn: Callable[[], None], *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) / iterations)


def _sdpa_call(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    force_flash: bool,
) -> torch.Tensor:
    context = nullcontext()
    if force_flash and sdpa_kernel is not None and SDPBackend is not None:
        context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    with context:
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
            dropout_p=0.0,
            enable_gqa=True,
        ).transpose(1, 2)


def _select_sdpa_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[str, bool]:
    try:
        _sdpa_call(q, k, v, force_flash=True)
        torch.cuda.synchronize()
        return "torch_flash_sdpa_gqa", True
    except (RuntimeError, TypeError):
        _sdpa_call(q, k, v, force_flash=False)
        torch.cuda.synchronize()
        return "torch_sdpa_gqa_auto", False


def _profile_cell(
    *,
    batch: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    q = torch.randn(batch, seq_len, q_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(batch, seq_len, kv_heads, head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    module = FusedOnlineAttention(
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        device=q.device,
        dtype=dtype,
    ).eval()
    baseline_name, force_flash = _select_sdpa_backend(q, k, v)

    with torch.no_grad():
        stream_output = module(q, k, v, causal=True)
        baseline_output = _sdpa_call(q, k, v, force_flash=force_flash)
        max_abs_error = float((stream_output - baseline_output).abs().max().item())
        stream_prefill_ms = _elapsed_ms(
            lambda: module(q, k, v, causal=True),
            warmup=warmup,
            iterations=iterations,
        )
        baseline_prefill_ms = _elapsed_ms(
            lambda: _sdpa_call(q, k, v, force_flash=force_flash),
            warmup=warmup,
            iterations=iterations,
        )

    module.train()
    qt = q.detach().clone().requires_grad_(True)
    kt = k.detach().clone().requires_grad_(True)
    vt = v.detach().clone().requires_grad_(True)
    grad = torch.randn_like(qt)

    def stream_train_step() -> None:
        qt.grad = kt.grad = vt.grad = None
        module(qt, kt, vt, causal=True).backward(grad)

    qr = q.detach().clone().requires_grad_(True)
    kr = k.detach().clone().requires_grad_(True)
    vr = v.detach().clone().requires_grad_(True)

    def baseline_train_step() -> None:
        qr.grad = kr.grad = vr.grad = None
        _sdpa_call(qr, kr, vr, force_flash=force_flash).backward(grad)

    stream_train_ms = _elapsed_ms(
        stream_train_step,
        warmup=warmup,
        iterations=iterations,
    )
    baseline_train_ms = _elapsed_ms(
        baseline_train_step,
        warmup=warmup,
        iterations=iterations,
    )

    return {
        "batch": batch,
        "seq_len": seq_len,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "group_size": q_heads // kv_heads,
        "head_dim": head_dim,
        "dtype": str(dtype).removeprefix("torch."),
        "baseline": baseline_name,
        "max_abs_error": max_abs_error,
        "stream_prefill_ms": stream_prefill_ms,
        "baseline_prefill_ms": baseline_prefill_ms,
        "prefill_speedup": baseline_prefill_ms / stream_prefill_ms,
        "stream_train_ms": stream_train_ms,
        "baseline_train_ms": baseline_train_ms,
        "train_speedup": baseline_train_ms / stream_train_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.q_heads % args.kv_heads:
        raise ValueError("q_heads must be a multiple of kv_heads")
    dtype = getattr(torch, args.dtype)
    rows = []
    for batch in args.batches:
        for seq_len in args.seq_lens:
            row = _profile_cell(
                batch=batch,
                seq_len=seq_len,
                q_heads=args.q_heads,
                kv_heads=args.kv_heads,
                head_dim=args.head_dim,
                dtype=dtype,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "rows": rows,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
