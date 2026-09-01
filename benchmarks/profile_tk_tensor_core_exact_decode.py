"""ThunderKittens tensor-core exact true-GQA decode baseline.

This spike answers one narrow backend question after the scalar TK head-mode
prototype: can a small true-GQA decode group use TK MMA instead of a scalar
per-head loop?

It intentionally starts with a compact, testable shape:

* batched M=1 decode;
* Q heads are packed by KV group into a 16-row tensor-core tile;
* K/V are packed as [B, Hkv, N, D] for the spike;
* one warp processes one KV group exactly.

This is not the final scheduler.  It is the exact-branch floor we need before
adding row masks and seed-only tile gating.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.profile_gate0_true_gqa import _dense_true_gqa  # noqa: E402
from benchmarks.profile_head_mode_decode_cuda import _torch_head_mode_reference  # noqa: E402
from benchmarks.profile_stream_attn_gate0_wrapper import _dtype, _error, _time_cuda  # noqa: E402
from benchmarks.profile_thunderkittens_extension_smoke import (  # noqa: E402
    _clone_tk,
    _find_tk_root,
    _tk_arch_define,
)

try:
    import flashinfer
except Exception:  # pragma: no cover - optional benchmark dependency
    flashinfer = None


def _make_flashinfer_batched_exact_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    page_size: int,
    workspace_mb: int,
):
    """Prepare FlashInfer's production batched paged-decode path once."""
    if flashinfer is None:
        raise RuntimeError("FlashInfer is not available")
    if k.shape != v.shape or k.ndim != 4:
        raise ValueError(f"expected matching [B,N,Hkv,D] K/V, got {k.shape=} {v.shape=}")
    batch, kv_len, kv_heads, dim = k.shape
    if q.ndim != 3 or q.shape[0] != batch or q.shape[2] != dim:
        raise ValueError(f"Q shape is incompatible with K/V: {q.shape=} {k.shape=}")
    if page_size <= 0 or workspace_mb <= 0:
        raise ValueError("FlashInfer page size and workspace size must be positive")

    pages_per_request = math.ceil(kv_len / page_size)
    padded_len = pages_per_request * page_size
    if padded_len == kv_len:
        k_padded = k
        v_padded = v
    else:
        k_padded = torch.zeros(
            batch, padded_len, kv_heads, dim, device=k.device, dtype=k.dtype
        )
        v_padded = torch.zeros_like(k_padded)
        k_padded[:, :kv_len].copy_(k)
        v_padded[:, :kv_len].copy_(v)

    key_pages = k_padded.view(batch * pages_per_request, page_size, kv_heads, dim)
    value_pages = v_padded.view(batch * pages_per_request, page_size, kv_heads, dim)
    paged_cache = torch.stack((key_pages, value_pages), dim=1).contiguous()
    total_pages = batch * pages_per_request
    indptr = torch.arange(
        0,
        total_pages + 1,
        pages_per_request,
        device=q.device,
        dtype=torch.int32,
    )
    indices = torch.arange(total_pages, device=q.device, dtype=torch.int32)
    last_page_len = torch.full(
        (batch,),
        kv_len - (pages_per_request - 1) * page_size,
        device=q.device,
        dtype=torch.int32,
    )
    workspace = torch.empty(
        workspace_mb * 1024 * 1024, device=q.device, dtype=torch.uint8
    )
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_tensor_cores=True,
        backend="auto",
    )
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        q.shape[1],
        kv_heads,
        dim,
        page_size,
        pos_encoding_mode="NONE",
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
        o_data_type=q.dtype,
        sm_scale=1.0 / math.sqrt(float(dim)),
        disable_split_kv=False,
    )
    out = torch.empty_like(q)

    def run() -> torch.Tensor:
        return wrapper.run(q, paged_cache, out=out)

    return run


from stream_attention.backends.sm80.tk_grouped_exact_sources import (
    CPP_SOURCE,
    CUDA_SOURCE,
)

def _compile_extension(
    *,
    tk_root: Path,
    cuda_arch: str,
    torch_cuda_arch_list: str,
    verbose: bool = False,
):
    from torch.utils.cpp_extension import load_inline

    previous_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = torch_cuda_arch_list
    try:
        build_dir = tempfile.mkdtemp(prefix="streamattn_tk_tc_exact_decode_")
        return load_inline(
            name="streamattn_tk_tc_exact_decode",
            cpp_sources=CPP_SOURCE,
            cuda_sources=CUDA_SOURCE,
            build_directory=build_dir,
            verbose=verbose,
            with_cuda=True,
            extra_include_paths=[str(tk_root / "include")],
            extra_cflags=["-std=c++20"],
            extra_cuda_cflags=[
                "-std=c++20",
                "-O3",
                "--use_fast_math",
                "--expt-relaxed-constexpr",
                "--expt-extended-lambda",
                f"-D{_tk_arch_define(cuda_arch)}",
            ],
        )
    finally:
        if previous_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous_arch


def _pack_q_by_kv_group(q: torch.Tensor, kv_heads: int, padded_rows: int = 16) -> torch.Tensor:
    if q.dim() != 3:
        raise ValueError("q must have shape [B,Hq,D]")
    batch, q_heads, dim = q.shape
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    group_size = q_heads // kv_heads
    if group_size > padded_rows:
        raise ValueError("group_size exceeds padded_rows")
    packed = torch.zeros((batch, kv_heads, padded_rows, dim), device=q.device, dtype=q.dtype)
    for kv_head in range(kv_heads):
        start = kv_head * group_size
        end = start + group_size
        packed[:, kv_head, :group_size, :] = q[:, start:end, :]
    return packed.contiguous()


def _unpack_q_by_kv_group(packed: torch.Tensor, q_heads: int) -> torch.Tensor:
    batch, kv_heads, _, dim = packed.shape
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    group_size = q_heads // kv_heads
    out = torch.empty((batch, q_heads, dim), device=packed.device, dtype=packed.dtype)
    for kv_head in range(kv_heads):
        start = kv_head * group_size
        end = start + group_size
        out[:, start:end, :] = packed[:, kv_head, :group_size, :]
    return out


def _pack_kv_head_major(kv: torch.Tensor) -> torch.Tensor:
    if kv.dim() != 4:
        raise ValueError("kv must have shape [B,N,Hkv,D]")
    return kv.permute(0, 2, 1, 3).contiguous()


def _parse_heads(raw: str) -> list[int]:
    return sorted(set(int(item.strip()) for item in raw.split(",") if item.strip()))


def _pack_row_modes_by_kv_group(
    *,
    q_heads: int,
    kv_heads: int,
    seed_heads: list[int],
    padded_rows: int = 16,
    device: torch.device,
) -> torch.Tensor:
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    group_size = q_heads // kv_heads
    if group_size > padded_rows:
        raise ValueError("group_size exceeds padded_rows")
    seed_set = set(seed_heads)
    modes = torch.full((kv_heads, padded_rows), 2, device=device, dtype=torch.int32)
    for kv_head in range(kv_heads):
        for row in range(group_size):
            q_head = kv_head * group_size + row
            modes[kv_head, row] = 1 if q_head in seed_set else 0
    return modes.contiguous()


def _tile_is_seed(
    *,
    tile: int,
    kv_len: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
) -> bool:
    token_start = tile * 16
    block_size = block_size or 16
    num_blocks = (kv_len + block_size - 1) // block_size
    sink_end = min(sink_blocks * block_size, kv_len)
    recent_start = 0 if recent_blocks >= num_blocks else (num_blocks - recent_blocks) * block_size
    keep = token_start < sink_end or token_start >= recent_start
    if middle_seed_blocks > 0:
        middle_seed_tokens = middle_seed_blocks * block_size
        if block_order == "sequential":
            middle_start = sink_end
            middle_end = min(middle_start + middle_seed_tokens, recent_start)
        else:
            middle_end = recent_start
            middle_start = max(sink_end, middle_end - middle_seed_tokens)
        keep = keep or (middle_start <= token_start < middle_end)
    return keep


def _pack_active_chunks_by_kv_group(
    *,
    q_heads: int,
    kv_heads: int,
    seed_heads: list[int],
    kv_len: int,
    num_chunks: int,
    block_size: int,
    sink_blocks: int,
    recent_blocks: int,
    middle_seed_blocks: int,
    block_order: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
    if q_heads % kv_heads != 0:
        raise ValueError("q_heads must be divisible by kv_heads")
    tiles = kv_len // 16
    if tiles % num_chunks != 0:
        raise ValueError("num_chunks must divide kv_len/16")
    group_size = q_heads // kv_heads
    seed_set = set(seed_heads)
    tiles_per_chunk = tiles // num_chunks
    active_by_kv: list[list[int]] = []
    for kv_head in range(kv_heads):
        q_start = kv_head * group_size
        q_group = range(q_start, q_start + group_size)
        has_exact = any(q_head not in seed_set for q_head in q_group)
        has_seed = any(q_head in seed_set for q_head in q_group)
        if has_exact:
            active = list(range(num_chunks))
        elif has_seed:
            active = [
                chunk
                for chunk in range(num_chunks)
                if any(
                    _tile_is_seed(
                        tile=tile,
                        kv_len=kv_len,
                        block_size=block_size,
                        sink_blocks=sink_blocks,
                        recent_blocks=recent_blocks,
                        middle_seed_blocks=middle_seed_blocks,
                        block_order=block_order,
                    )
                    for tile in range(chunk * tiles_per_chunk, (chunk + 1) * tiles_per_chunk)
                )
            ]
        else:
            active = []
        active_by_kv.append(active)

    max_active = max((len(chunks) for chunks in active_by_kv), default=0)
    if max_active == 0:
        max_active = 1
    active_chunks = torch.zeros((kv_heads, max_active), device=device, dtype=torch.int32)
    active_counts = torch.empty((kv_heads,), device=device, dtype=torch.int32)
    flat_chunks: list[int] = []
    offsets = [0]
    for kv_head, chunks in enumerate(active_by_kv):
        active_counts[kv_head] = len(chunks)
        if chunks:
            active_chunks[kv_head, : len(chunks)] = torch.tensor(chunks, device=device, dtype=torch.int32)
            flat_chunks.extend(chunks)
        offsets.append(len(flat_chunks))
    if not flat_chunks:
        flat_chunks = [0]
    flat_active_chunks = torch.tensor(flat_chunks, device=device, dtype=torch.int32)
    active_offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    return (
        active_chunks.contiguous(),
        active_counts.contiguous(),
        flat_active_chunks.contiguous(),
        active_offsets.contiguous(),
        active_by_kv,
    )


def _reference_from_packed(q_group: torch.Tensor, k_group: torch.Tensor, v_group: torch.Tensor) -> torch.Tensor:
    # Reference over the padded rows; caller can unpack actual Q heads.
    batch, kv_heads, padded_rows, dim = q_group.shape
    outputs = []
    scale = dim**-0.5
    for kv_head in range(kv_heads):
        qh = q_group[:, kv_head, :, :].float()
        kh = k_group[:, kv_head, :, :].float()
        vh = v_group[:, kv_head, :, :].float()
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(probs, vh).to(q_group.dtype))
    return torch.stack(outputs, dim=1).contiguous()


def _find_or_clone_tk(args: argparse.Namespace) -> Path:
    tk_root = _find_tk_root(args.tk_root)
    if tk_root is not None:
        return tk_root
    if not args.checkout_dir:
        raise RuntimeError("ThunderKittens root not found; pass --tk-root or --checkout-dir")
    clone = _clone_tk(Path(args.checkout_dir).expanduser() / "ThunderKittens")
    tk_root = _find_tk_root(str(Path(args.checkout_dir).expanduser() / "ThunderKittens"))
    if tk_root is None:
        raise RuntimeError(f"ThunderKittens clone failed: {clone}")
    return tk_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--kv-len", type=int, default=32768)
    parser.add_argument("--q-heads", type=int, default=14)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--num-chunks", type=int, default=64)
    parser.add_argument("--num-chunks-list", default="")
    parser.add_argument("--producer-warps-list", default="1,2,4,8")
    parser.add_argument("--seed-heads", default="2,3,4,6,7")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--sink-blocks", type=int, default=2)
    parser.add_argument("--recent-blocks", type=int, default=2)
    parser.add_argument("--middle-seed-blocks", type=int, default=2)
    parser.add_argument("--block-order", default="recent_first", choices=["sequential", "recent_first"])
    parser.add_argument("--flashinfer-page-size", type=int, default=16)
    parser.add_argument("--flashinfer-workspace-mb", type=int, default=128)
    parser.add_argument("--tk-root", default="")
    parser.add_argument("--checkout-dir", default="")
    parser.add_argument("--cuda-arch", default="sm_90a")
    parser.add_argument("--torch-cuda-arch-list", default="9.0a")
    parser.add_argument("--compile-verbose", action="store_true")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    if args.dtype != "bf16":
        raise ValueError("this spike currently supports --dtype bf16 only")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.head_dim not in (64, 128):
        raise ValueError("this spike currently supports --head-dim 64 or 128 only")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    torch.manual_seed(args.seed)
    q = torch.randn((args.batch, args.q_heads, args.head_dim), device=device, dtype=dtype)
    k = torch.randn(
        (args.batch, args.kv_len, args.kv_heads, args.head_dim),
        device=device,
        dtype=dtype,
    )
    v = torch.randn_like(k)
    q_group = _pack_q_by_kv_group(q, args.kv_heads, padded_rows=16)
    k_group = _pack_kv_head_major(k)
    v_group = _pack_kv_head_major(v)
    group_size = args.q_heads // args.kv_heads
    q_runtime_view = q.view(args.batch, args.kv_heads, group_size, args.head_dim)
    runtime_io_output = torch.empty_like(q)
    runtime_io_output_view = runtime_io_output.view(
        args.batch, args.kv_heads, group_size, args.head_dim
    )
    seed_heads = _parse_heads(args.seed_heads)
    row_modes = _pack_row_modes_by_kv_group(
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        seed_heads=seed_heads,
        padded_rows=16,
        device=device,
    )
    active_chunks, active_counts, flat_active_chunks, active_offsets, active_by_kv = _pack_active_chunks_by_kv_group(
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        seed_heads=seed_heads,
        kv_len=args.kv_len,
        num_chunks=args.num_chunks,
        block_size=args.block_size,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
        middle_seed_blocks=args.middle_seed_blocks,
        block_order=args.block_order,
        device=device,
    )
    block_order_id = 0 if args.block_order == "sequential" else 1

    tk_root = _find_or_clone_tk(args)
    print(
        "[tk-tc] compiling extension "
        f"batch={args.batch} head_dim={args.head_dim} dtype={args.dtype} q_heads={args.q_heads} "
        f"kv_heads={args.kv_heads} kv_len={args.kv_len}",
        flush=True,
    )
    compile_start = time.perf_counter()
    ext = _compile_extension(
        tk_root=tk_root,
        cuda_arch=args.cuda_arch,
        torch_cuda_arch_list=args.torch_cuda_arch_list,
        verbose=args.compile_verbose,
    )
    compile_s = time.perf_counter() - compile_start
    print(f"[tk-tc] compile finished in {compile_s:.2f}s", flush=True)

    def tk_exact() -> torch.Tensor:
        return ext.exact_decode(q_group, k_group, v_group)

    def _chunk_counts() -> list[int]:
        counts = [args.num_chunks]
        if args.num_chunks_list:
            counts.extend(int(item.strip()) for item in args.num_chunks_list.split(",") if item.strip())
        return sorted(set(counts))

    chunk_counts = _chunk_counts()
    direct_producer_warps = 4
    direct_workspaces: dict[
        int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = {}
    for count in chunk_counts:
        grouped_chunks = count // direct_producer_warps
        direct_workspaces[count] = (
            torch.empty(
                args.batch,
                args.kv_heads,
                grouped_chunks * 16,
                args.head_dim,
                device=device,
                dtype=dtype,
            ),
            torch.empty(
                args.batch,
                args.kv_heads,
                grouped_chunks,
                16,
                device=device,
                dtype=torch.float32,
            ),
            torch.empty_like(q),
        )
    producer_warp_counts = sorted(
        set(int(item.strip()) for item in args.producer_warps_list.split(",") if item.strip())
    )
    invalid_warp_counts = [count for count in producer_warp_counts if count not in (1, 2, 4, 8)]
    if invalid_warp_counts:
        raise ValueError(f"producer warp counts must be in (1,2,4,8), got {invalid_warp_counts}")
    compact_inputs_by_chunk: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]] = {}
    for count in chunk_counts:
        compact_inputs_by_chunk[count] = _pack_active_chunks_by_kv_group(
            q_heads=args.q_heads,
            kv_heads=args.kv_heads,
            seed_heads=seed_heads,
            kv_len=args.kv_len,
            num_chunks=count,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
            device=device,
        )

    def tk_chunk_only(num_chunks: int) -> torch.Tensor:
        return ext.exact_decode_chunks(q_group, k_group, v_group, num_chunks)

    def tk_chunk_merged(num_chunks: int) -> torch.Tensor:
        return ext.exact_decode_chunk_merged(q_group, k_group, v_group, num_chunks)

    def tk_chunk_warpgroup(num_chunks: int, producer_warps: int) -> torch.Tensor:
        return ext.exact_decode_chunk_states_warpgroup(
            q_group, k_group, v_group, num_chunks, producer_warps
        )[0]

    def tk_merged_warpgroup(num_chunks: int, producer_warps: int) -> torch.Tensor:
        return ext.exact_decode_chunk_merged_warpgroup(
            q_group, k_group, v_group, num_chunks, producer_warps
        )

    def tk_chunk_staged(num_chunks: int) -> torch.Tensor:
        return ext.exact_decode_chunk_states_staged(
            q_group, k_group, v_group, num_chunks
        )[0]

    def tk_merged_staged(num_chunks: int) -> torch.Tensor:
        return ext.exact_decode_chunk_merged_staged(
            q_group, k_group, v_group, num_chunks
        )

    def tk_chunk_staged_grouped(num_chunks: int) -> torch.Tensor:
        return ext.exact_decode_chunk_states_staged_grouped(
            q_group, k_group, v_group, num_chunks
        )[0]

    def tk_merged_staged_grouped(num_chunks: int) -> torch.Tensor:
        return ext.exact_decode_chunk_merged_staged_grouped(
            q_group, k_group, v_group, num_chunks
        )

    def tk_merged_staged_grouped_runtime_io(num_chunks: int) -> torch.Tensor:
        q_group[:, :, :group_size, :].copy_(q_runtime_view)
        grouped_output = ext.exact_decode_chunk_merged_staged_grouped(
            q_group, k_group, v_group, num_chunks
        )
        runtime_io_output_view.copy_(grouped_output[:, :, :group_size, :])
        return runtime_io_output

    def tk_merged_staged_grouped_direct(num_chunks: int) -> torch.Tensor:
        return ext.exact_decode_chunk_merged_staged_grouped_direct(
            q, k_group, v_group, num_chunks
        )

    def tk_merged_staged_grouped_direct_out(num_chunks: int) -> torch.Tensor:
        partial_out, partial_lse, direct_out = direct_workspaces[num_chunks]
        return ext.exact_decode_chunk_merged_staged_grouped_direct_out(
            q,
            k_group,
            v_group,
            partial_out,
            partial_lse,
            direct_out,
            num_chunks,
        )

    def tk_head_mode_merged(num_chunks: int) -> torch.Tensor:
        return ext.head_mode_chunk_merged(
            q_group,
            k_group,
            v_group,
            row_modes,
            num_chunks,
            args.block_size,
            args.sink_blocks,
            args.recent_blocks,
            args.middle_seed_blocks,
            block_order_id,
        )

    def tk_head_mode_compact(num_chunks: int) -> torch.Tensor:
        compact_chunks, compact_counts, compact_flat_chunks, compact_offsets, _ = compact_inputs_by_chunk[num_chunks]
        return ext.head_mode_compact_chunk_merged(
            q_group,
            k_group,
            v_group,
            row_modes,
            compact_chunks,
            compact_counts,
            compact_flat_chunks,
            compact_offsets,
            num_chunks,
            args.block_size,
            args.sink_blocks,
            args.recent_blocks,
            args.middle_seed_blocks,
            block_order_id,
        )

    print("[tk-tc] running correctness references", flush=True)
    tk_out_group = tk_exact()
    partial_group = tk_chunk_only(args.num_chunks)
    merged_group = tk_chunk_merged(args.num_chunks)
    warpgroup_outputs = {
        str(producer_warps): tk_merged_warpgroup(args.num_chunks, producer_warps)
        for producer_warps in producer_warp_counts
    }
    staged_merged_group = tk_merged_staged(args.num_chunks)
    staged_grouped_merged_group = (
        tk_merged_staged_grouped(args.num_chunks) if args.head_dim == 64 else None
    )
    staged_grouped_runtime_io_out = (
        tk_merged_staged_grouped_runtime_io(args.num_chunks)
        if args.head_dim == 64
        else None
    )
    staged_grouped_direct_out = tk_merged_staged_grouped_direct(args.num_chunks)
    staged_grouped_direct_preallocated_out = tk_merged_staged_grouped_direct_out(
        args.num_chunks
    )
    head_mode_group = tk_head_mode_merged(args.num_chunks)
    compact_head_mode_group = ext.head_mode_compact_chunk_merged(
        q_group,
        k_group,
        v_group,
        row_modes,
        active_chunks,
        active_counts,
        flat_active_chunks,
        active_offsets,
        args.num_chunks,
        args.block_size,
        args.sink_blocks,
        args.recent_blocks,
        args.middle_seed_blocks,
        block_order_id,
    )
    torch_ref_group = _reference_from_packed(q_group, k_group, v_group)
    tk_out = _unpack_q_by_kv_group(tk_out_group, args.q_heads)
    merged_out = _unpack_q_by_kv_group(merged_group, args.q_heads)
    head_mode_out = _unpack_q_by_kv_group(head_mode_group, args.q_heads)
    compact_head_mode_out = _unpack_q_by_kv_group(compact_head_mode_group, args.q_heads)
    staged_merged_out = _unpack_q_by_kv_group(staged_merged_group, args.q_heads)
    staged_grouped_merged_out = (
        _unpack_q_by_kv_group(staged_grouped_merged_group, args.q_heads)
        if staged_grouped_merged_group is not None
        else None
    )

    def dense_true() -> torch.Tensor:
        return _dense_true_gqa(q[:, None, :, :], k, v)[:, 0]

    head_modes = torch.zeros(args.q_heads, device=device, dtype=torch.int32)
    if seed_heads:
        head_modes[torch.tensor(seed_heads, device=device, dtype=torch.long)] = 1

    def head_mode_ref() -> torch.Tensor:
        return _torch_head_mode_reference(
            q,
            k,
            v,
            head_modes,
            block_size=args.block_size,
            sink_blocks=args.sink_blocks,
            recent_blocks=args.recent_blocks,
            middle_seed_blocks=args.middle_seed_blocks,
            block_order=args.block_order,
        )

    dense_ref = dense_true()
    head_ref = head_mode_ref()
    print("[tk-tc] timing kernels", flush=True)
    flashinfer_out = None
    try:
        flashinfer_run = _make_flashinfer_batched_exact_runner(
            q,
            k,
            v,
            page_size=args.flashinfer_page_size,
            workspace_mb=args.flashinfer_workspace_mb,
        )
        flashinfer_out = flashinfer_run().clone()
        flashinfer_ms = _time_cuda(
            flashinfer_run,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
    except Exception as exc:  # pragma: no cover - depends on optional backend
        flashinfer_ms = None
        flashinfer_error = str(exc)
    else:
        flashinfer_error = None

    tk_ms = _time_cuda(tk_exact, device=device, warmup=args.warmup, iters=args.iters)
    chunk_sweep: Dict[str, float] = {}
    merged_sweep: Dict[str, float] = {}
    head_mode_sweep: Dict[str, float] = {}
    compact_head_mode_sweep: Dict[str, float] = {}
    warpgroup_chunk_sweep: Dict[str, float] = {}
    warpgroup_merged_sweep: Dict[str, float] = {}
    staged_chunk_sweep: Dict[str, float] = {}
    staged_merged_sweep: Dict[str, float] = {}
    staged_grouped_chunk_sweep: Dict[str, float] = {}
    staged_grouped_merged_sweep: Dict[str, float] = {}
    staged_grouped_runtime_io_sweep: Dict[str, float] = {}
    staged_grouped_direct_sweep: Dict[str, float] = {}
    staged_grouped_direct_out_sweep: Dict[str, float] = {}
    for num_chunks in chunk_counts:
        chunk_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_chunk_only(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        merged_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_chunk_merged(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        head_mode_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_head_mode_merged(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        compact_head_mode_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_head_mode_compact(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        staged_chunk_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_chunk_staged(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        staged_merged_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_merged_staged(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        if args.head_dim == 64:
            staged_grouped_chunk_sweep[str(num_chunks)] = _time_cuda(
                lambda c=num_chunks: tk_chunk_staged_grouped(c),
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
            staged_grouped_merged_sweep[str(num_chunks)] = _time_cuda(
                lambda c=num_chunks: tk_merged_staged_grouped(c),
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
            staged_grouped_runtime_io_sweep[str(num_chunks)] = _time_cuda(
                lambda c=num_chunks: tk_merged_staged_grouped_runtime_io(c),
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
        staged_grouped_direct_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_merged_staged_grouped_direct(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        staged_grouped_direct_out_sweep[str(num_chunks)] = _time_cuda(
            lambda c=num_chunks: tk_merged_staged_grouped_direct_out(c),
            device=device,
            warmup=args.warmup,
            iters=args.iters,
        )
        for producer_warps in producer_warp_counts:
            key = f"c{num_chunks}_w{producer_warps}"
            warpgroup_chunk_sweep[key] = _time_cuda(
                lambda c=num_chunks, w=producer_warps: tk_chunk_warpgroup(c, w),
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
            warpgroup_merged_sweep[key] = _time_cuda(
                lambda c=num_chunks, w=producer_warps: tk_merged_warpgroup(c, w),
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
    chunk_ms = chunk_sweep[str(args.num_chunks)]
    merged_ms = merged_sweep[str(args.num_chunks)]
    dense_ms = _time_cuda(dense_true, device=device, warmup=args.warmup, iters=args.iters)
    output = {
        "schema": "streamattn.tk_tensor_core_exact_decode.v1",
        "shape": {
            "batch": args.batch,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "group_size": args.q_heads // args.kv_heads,
            "padded_group_rows": 16,
            "num_chunks": args.num_chunks,
            "seed_heads": seed_heads,
            "block_size": args.block_size,
            "seed_tile_blocks": {
                "sink_blocks": args.sink_blocks,
                "recent_blocks": args.recent_blocks,
                "middle_seed_blocks": args.middle_seed_blocks,
                "block_order": args.block_order,
            },
            "active_chunks_by_kv_group": active_by_kv,
            "active_chunk_counts_by_kv_group": [len(chunks) for chunks in active_by_kv],
            "kv_len": args.kv_len,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "kv_layout_runtime": "B,Hkv,N,D",
            "note": "spike packs NHD KV to head-major layout before the TK kernel",
        },
        "compile": {
            "tk_root": str(tk_root),
            "compile_s": compile_s,
            "cuda_arch": args.cuda_arch,
        },
        "timing": {
            "tk_tensor_core_exact_ms": tk_ms,
            "tk_tensor_core_chunk_only_ms": chunk_ms,
            "tk_tensor_core_chunk_merged_ms": merged_ms,
            "tk_tensor_core_chunk_only_sweep_ms": chunk_sweep,
            "tk_tensor_core_chunk_merged_sweep_ms": merged_sweep,
            "tk_tensor_core_head_mode_merged_sweep_ms": head_mode_sweep,
            "tk_tensor_core_head_mode_compact_sweep_ms": compact_head_mode_sweep,
            "tk_tensor_core_warpgroup_chunk_sweep_ms": warpgroup_chunk_sweep,
            "tk_tensor_core_warpgroup_merged_sweep_ms": warpgroup_merged_sweep,
            "tk_tensor_core_staged_chunk_sweep_ms": staged_chunk_sweep,
            "tk_tensor_core_staged_merged_sweep_ms": staged_merged_sweep,
            "tk_tensor_core_staged_grouped_chunk_sweep_ms": staged_grouped_chunk_sweep,
            "tk_tensor_core_staged_grouped_merged_sweep_ms": staged_grouped_merged_sweep,
            "tk_tensor_core_staged_grouped_runtime_io_sweep_ms": staged_grouped_runtime_io_sweep,
            "tk_tensor_core_staged_grouped_direct_sweep_ms": staged_grouped_direct_sweep,
            "tk_tensor_core_staged_grouped_direct_out_sweep_ms": staged_grouped_direct_out_sweep,
            "tk_tensor_core_staged_grouped_direct_producer_warps": direct_producer_warps,
            "tk_tensor_core_best_staged_grouped_chunk_ms": min(
                staged_grouped_chunk_sweep.values()
            )
            if staged_grouped_chunk_sweep
            else None,
            "tk_tensor_core_best_staged_grouped_chunk_count": int(
                min(staged_grouped_chunk_sweep, key=staged_grouped_chunk_sweep.get)
            )
            if staged_grouped_chunk_sweep
            else None,
            "tk_tensor_core_best_staged_grouped_merged_ms": min(
                staged_grouped_merged_sweep.values()
            )
            if staged_grouped_merged_sweep
            else None,
            "tk_tensor_core_best_staged_grouped_merged_chunk_count": int(
                min(staged_grouped_merged_sweep, key=staged_grouped_merged_sweep.get)
            )
            if staged_grouped_merged_sweep
            else None,
            "tk_tensor_core_best_staged_grouped_runtime_io_ms": min(
                staged_grouped_runtime_io_sweep.values()
            )
            if staged_grouped_runtime_io_sweep
            else None,
            "tk_tensor_core_best_staged_grouped_runtime_io_chunk_count": int(
                min(staged_grouped_runtime_io_sweep, key=staged_grouped_runtime_io_sweep.get)
            )
            if staged_grouped_runtime_io_sweep
            else None,
            "tk_tensor_core_best_staged_grouped_direct_ms": min(
                staged_grouped_direct_sweep.values()
            )
            if staged_grouped_direct_sweep
            else None,
            "tk_tensor_core_best_staged_grouped_direct_chunk_count": int(
                min(staged_grouped_direct_sweep, key=staged_grouped_direct_sweep.get)
            )
            if staged_grouped_direct_sweep
            else None,
            "tk_tensor_core_best_staged_grouped_direct_out_ms": min(
                staged_grouped_direct_out_sweep.values()
            )
            if staged_grouped_direct_out_sweep
            else None,
            "tk_tensor_core_best_staged_grouped_direct_out_chunk_count": int(
                min(staged_grouped_direct_out_sweep, key=staged_grouped_direct_out_sweep.get)
            )
            if staged_grouped_direct_out_sweep
            else None,
            "tk_tensor_core_best_staged_chunk_ms": min(staged_chunk_sweep.values())
            if staged_chunk_sweep
            else None,
            "tk_tensor_core_best_staged_chunk_count": int(
                min(staged_chunk_sweep, key=staged_chunk_sweep.get)
            )
            if staged_chunk_sweep
            else None,
            "tk_tensor_core_best_staged_merged_ms": min(staged_merged_sweep.values())
            if staged_merged_sweep
            else None,
            "tk_tensor_core_best_staged_merged_chunk_count": int(
                min(staged_merged_sweep, key=staged_merged_sweep.get)
            )
            if staged_merged_sweep
            else None,
            "tk_tensor_core_best_warpgroup_chunk_ms": min(warpgroup_chunk_sweep.values())
            if warpgroup_chunk_sweep
            else None,
            "tk_tensor_core_best_warpgroup_chunk_config": min(
                warpgroup_chunk_sweep, key=warpgroup_chunk_sweep.get
            )
            if warpgroup_chunk_sweep
            else None,
            "tk_tensor_core_best_warpgroup_merged_ms": min(warpgroup_merged_sweep.values())
            if warpgroup_merged_sweep
            else None,
            "tk_tensor_core_best_warpgroup_merged_config": min(
                warpgroup_merged_sweep, key=warpgroup_merged_sweep.get
            )
            if warpgroup_merged_sweep
            else None,
            "tk_tensor_core_best_chunk_only_ms": min(chunk_sweep.values()) if chunk_sweep else None,
            "tk_tensor_core_best_chunk_count": int(min(chunk_sweep, key=chunk_sweep.get)) if chunk_sweep else None,
            "tk_tensor_core_best_merged_ms": min(merged_sweep.values()) if merged_sweep else None,
            "tk_tensor_core_best_merged_chunk_count": int(min(merged_sweep, key=merged_sweep.get))
            if merged_sweep
            else None,
            "tk_tensor_core_best_head_mode_ms": min(head_mode_sweep.values()) if head_mode_sweep else None,
            "tk_tensor_core_best_head_mode_chunk_count": int(min(head_mode_sweep, key=head_mode_sweep.get))
            if head_mode_sweep
            else None,
            "tk_tensor_core_best_head_mode_compact_ms": min(compact_head_mode_sweep.values())
            if compact_head_mode_sweep
            else None,
            "tk_tensor_core_best_head_mode_compact_chunk_count": int(
                min(compact_head_mode_sweep, key=compact_head_mode_sweep.get)
            )
            if compact_head_mode_sweep
            else None,
            "torch_dense_true_gqa_ms": dense_ms,
            "flashinfer_exact_ms": flashinfer_ms,
            "flashinfer_baseline": "batch_decode_paged_nhd_auto_tensor_cores",
            "flashinfer_page_size": args.flashinfer_page_size,
            "tk_speedup_vs_torch_dense": dense_ms / tk_ms if tk_ms else None,
            "tk_speedup_vs_flashinfer": flashinfer_ms / tk_ms if flashinfer_ms and tk_ms else None,
            "chunk_only_speedup_vs_tk_serial": tk_ms / chunk_ms if chunk_ms else None,
            "chunk_only_speedup_vs_flashinfer": flashinfer_ms / chunk_ms if flashinfer_ms and chunk_ms else None,
        },
        "quality": {
            "tk_vs_packed_torch_ref": _error(tk_out_group, torch_ref_group),
            "tk_vs_dense_true_gqa": _error(tk_out[:, None, :, :], dense_ref[:, None, :, :]),
            "merged_vs_packed_torch_ref": _error(merged_group, torch_ref_group),
            "merged_vs_dense_true_gqa": _error(merged_out[:, None, :, :], dense_ref[:, None, :, :]),
            "staged_merged_vs_packed_torch_ref": _error(staged_merged_group, torch_ref_group),
            "staged_merged_vs_dense_true_gqa": _error(
                staged_merged_out[:, None, :, :], dense_ref[:, None, :, :]
            ),
            "staged_grouped_merged_vs_packed_torch_ref": (
                _error(staged_grouped_merged_group, torch_ref_group)
                if staged_grouped_merged_group is not None
                else None
            ),
            "staged_grouped_merged_vs_dense_true_gqa": (
                _error(
                    staged_grouped_merged_out[:, None, :, :],
                    dense_ref[:, None, :, :],
                )
                if staged_grouped_merged_out is not None
                else None
            ),
            "staged_grouped_runtime_io_vs_dense_true_gqa": (
                _error(
                    staged_grouped_runtime_io_out[:, None, :, :],
                    dense_ref[:, None, :, :],
                )
                if staged_grouped_runtime_io_out is not None
                else None
            ),
            "staged_grouped_direct_vs_dense_true_gqa": _error(
                staged_grouped_direct_out[:, None, :, :], dense_ref[:, None, :, :]
            ),
            "staged_grouped_direct_out_vs_dense_true_gqa": _error(
                staged_grouped_direct_preallocated_out[:, None, :, :],
                dense_ref[:, None, :, :],
            ),
            "flashinfer_vs_dense_true_gqa": _error(
                flashinfer_out[:, None, :, :], dense_ref[:, None, :, :]
            )
            if flashinfer_out is not None
            else None,
            "warpgroup_merged_vs_packed_torch_ref": {
                producer_warps: _error(result, torch_ref_group)
                for producer_warps, result in warpgroup_outputs.items()
            },
            "head_mode_vs_reference": _error(head_mode_out, head_ref),
            "head_mode_vs_dense_true_gqa": _error(head_mode_out[:, None, :, :], dense_ref[:, None, :, :]),
            "compact_head_mode_vs_reference": _error(compact_head_mode_out, head_ref),
            "compact_head_mode_vs_dense_true_gqa": _error(
                compact_head_mode_out[:, None, :, :],
                dense_ref[:, None, :, :],
            ),
            "partial_group_shape": list(partial_group.shape),
        },
        "flashinfer_error": flashinfer_error,
        "next_path": "calibrate_kv_group_coherent_seed_policies_and_reduce_merge_overhead"
        if (flashinfer_ms is not None and min(merged_sweep.values()) <= flashinfer_ms * 1.5)
        else "optimize_partial_state_merge_or_embed_head_modes_in_flashinfer_scheduler",
    }
    text = json.dumps(output, indent=2, sort_keys=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
