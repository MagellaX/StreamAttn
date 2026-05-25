"""Serving-facing StreamAttn seed-only decode example.

This example demonstrates the first bounded StreamAttn wedge:

    Qwen2.5-0.5B, L8, 32K bucket, fp16, true GQA, batch >= 8.

The API is intentionally fail-closed.  If the policy does not match the tensors
or the seed-only backend is unavailable, the service uses the injected dense
fallback and reports the reason in ``StreamAttnServingInfo``.

CPU-safe smoke:

    python examples/seed_only_serving_decode.py --backend torch --batch 4 --kv-len 128 --dtype fp32

Real serving-shape smoke with FlashInfer fallback:

    python examples/seed_only_serving_decode.py --backend flashinfer --device cuda --batch 8 --kv-len 32768 --dtype fp16
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stream_attention import (  # noqa: E402
    StreamAttnDecodePolicy,
    StreamAttnSeedOnlyDecodeService,
    dense_attention_forward,
)


DenseFallback = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def make_torch_dense_fallback() -> DenseFallback:
    """Dense fallback with the repository's reference attention."""

    def fallback(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return dense_attention_forward(query, key, value, causal=False)

    return fallback


def _make_paged_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("expected matching [B, N, Hkv, D] key/value tensors")
    batch, kv_len, kv_heads, dim = key.shape
    pages_per_request = math.ceil(kv_len / page_size)
    total_pages = batch * pages_per_request
    cache = torch.zeros(
        total_pages,
        2,
        page_size,
        kv_heads,
        dim,
        device=key.device,
        dtype=key.dtype,
    )
    for batch_idx in range(batch):
        for local_page in range(pages_per_request):
            start = local_page * page_size
            end = min(kv_len, start + page_size)
            page = batch_idx * pages_per_request + local_page
            cache[page, 0, : end - start] = key[batch_idx, start:end]
            cache[page, 1, : end - start] = value[batch_idx, start:end]
    indptr = torch.arange(
        0,
        total_pages + 1,
        pages_per_request,
        device=key.device,
        dtype=torch.int32,
    )
    indices = torch.arange(total_pages, device=key.device, dtype=torch.int32)
    last_page_len = torch.full(
        (batch,),
        kv_len - (pages_per_request - 1) * page_size,
        device=key.device,
        dtype=torch.int32,
    )
    return cache, indptr, indices, last_page_len


def make_flashinfer_dense_fallback(
    *,
    page_size: int = 32,
    workspace_mb: int = 256,
    backend: str = "auto",
    use_tensor_cores: bool = True,
    disable_split_kv: bool = False,
) -> DenseFallback:
    """Build an injectable FlashInfer dense fallback.

    This compact example plans the FlashInfer wrapper per call.  A production
    server should reuse the workspace and plan lifecycle across decode steps.
    """

    try:
        import flashinfer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("FlashInfer is not installed") from exc

    def fallback(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if query.ndim != 4 or query.shape[1] != 1:
            raise ValueError("FlashInfer decode fallback expects query [B, 1, Hq, D]")
        kv_cache, indptr, indices, last_page_len = _make_paged_kv_cache(
            key,
            value,
            page_size=page_size,
        )
        workspace = torch.empty(
            workspace_mb * 1024 * 1024,
            device=query.device,
            dtype=torch.uint8,
        )
        wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            use_tensor_cores=use_tensor_cores,
            backend=backend,
        )
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            query.shape[2],
            key.shape[2],
            query.shape[3],
            page_size,
            pos_encoding_mode="NONE",
            q_data_type=query.dtype,
            kv_data_type=key.dtype,
            o_data_type=query.dtype,
            sm_scale=1.0 / math.sqrt(float(query.shape[3])),
            disable_split_kv=disable_split_kv,
        )
        return wrapper.run(query[:, 0].contiguous(), kv_cache).view_as(query)

    return fallback


def make_service(args: argparse.Namespace) -> StreamAttnSeedOnlyDecodeService:
    if args.backend == "flashinfer":
        dense_fallback = make_flashinfer_dense_fallback(
            page_size=args.page_size,
            workspace_mb=args.workspace_mb,
            backend=args.flashinfer_backend,
            use_tensor_cores=not args.no_flashinfer_tensor_cores,
            disable_split_kv=args.disable_split_kv,
        )
        fallback_backend = "flashinfer_dense"
    else:
        dense_fallback = make_torch_dense_fallback()
        fallback_backend = "torch_dense"

    return StreamAttnSeedOnlyDecodeService.from_packaged(
        dense_fallback=dense_fallback,
        dense_fallback_backend=fallback_backend,
        decode_policy=StreamAttnDecodePolicy(
            collect_telemetry_every=0,
            min_kv_len_for_gate0_seed_only=args.min_kv_len_for_seed_only,
            safety_margin=args.safety_margin,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["torch", "flashinfer"], default="torch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=128)
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=["auto", "dense"], default="auto")
    parser.add_argument("--min-kv-len-for-seed-only", type=int, default=16384)
    parser.add_argument("--safety-margin", type=float, default=1.10)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--flashinfer-backend", default="auto")
    parser.add_argument("--no-flashinfer-tensor-cores", action="store_true")
    parser.add_argument("--disable-split-kv", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    query = torch.randn(
        args.batch,
        1,
        args.q_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        args.batch,
        args.kv_len,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    value = torch.randn_like(key)

    service = make_service(args)
    output, info = service.run(query, key, value, mode=args.mode)

    print(
        json.dumps(
            {
                "output_shape": list(output.shape),
                "serving_info": info.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
