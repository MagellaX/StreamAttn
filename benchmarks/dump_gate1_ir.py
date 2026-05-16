"""Dump Triton IR/PTX for Gate-1 kernels.

Run this on a CUDA machine with Triton installed. It intentionally imports the
kernel objects directly so we can inspect generic versus mass-specialized codegen
without editing the benchmark harness.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import triton

from stream_attention.kernels.gate1_fwd_triton import _gate1_fwd_kernel
from stream_attention.kernels.gate1_mass_fwd_triton import _gate1_mass_fwd_kernel


def _make_tensors(args):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda")
    q = torch.zeros(args.batch, args.seq_q, args.heads, args.dim, device=device, dtype=dtype)
    k = torch.zeros(args.batch, args.seq_k, args.heads, args.dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q[..., 0] = args.peak
    active_tokens = min(args.seq_k, args.active_blocks * args.block_size)
    k[:, :active_tokens, :, 0] = args.peak
    k[:, active_tokens:, :, 0] = -args.peak
    return q, k, v


def _dump_generic(args, q, k, v, out_dir: Path):
    batch, seq_q, heads, dim = q.shape
    seq_k = k.shape[1]
    q_blocks = triton.cdiv(seq_q, args.tile_size_q)
    out = torch.empty_like(q)
    stats = torch.empty(batch, heads, q_blocks, 6, device=q.device, dtype=torch.int32)
    value_bounds = torch.empty(
        batch,
        heads,
        triton.cdiv(seq_k, args.block_size),
        device=q.device,
        dtype=torch.float32,
    )
    compiled = _gate1_fwd_kernel.warmup(
        q,
        k,
        v,
        value_bounds,
        out,
        stats,
        M=seq_q,
        N=seq_k,
        H=heads,
        D=dim,
        NUM_BLOCKS=triton.cdiv(seq_k, args.block_size),
        TILE_M=args.tile_size_q,
        TILE_N=args.block_size,
        SCALE=1.0 / math.sqrt(dim),
        ERROR_BUDGET=float(args.error_budget),
        LOG_ERROR_BUDGET=math.log(max(float(args.error_budget), 1.0e-20)),
        POST_QK_THRESHOLD=0.0,
        FORCE_MODE=args.force_mode,
        IS_CAUSAL=args.causal,
        VALUE_BOUND=args.skip_predicate == "value_bound",
        PV_USE_BF16=v.dtype is torch.bfloat16,
        HAS_STATS=True,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        grid=(q_blocks, batch, heads),
    )
    return _write_asm(compiled.asm, out_dir)


def _dump_mass(args, q, k, v, out_dir: Path):
    batch, seq_q, heads, dim = q.shape
    seq_k = k.shape[1]
    q_blocks = triton.cdiv(seq_q, args.tile_size_q)
    out = torch.empty_like(q)
    stats = torch.empty(batch, heads, q_blocks, 6, device=q.device, dtype=torch.int32)
    compiled = _gate1_mass_fwd_kernel.warmup(
        q,
        k,
        v,
        out,
        stats,
        M=seq_q,
        N=seq_k,
        H=heads,
        D=dim,
        NUM_BLOCKS=triton.cdiv(seq_k, args.block_size),
        TILE_M=args.tile_size_q,
        TILE_N=args.block_size,
        SCALE=1.0 / math.sqrt(dim),
        ERROR_BUDGET=float(args.error_budget),
        LOG_ERROR_BUDGET=math.log(max(float(args.error_budget), 1.0e-20)),
        POST_QK_THRESHOLD=0.0,
        FORCE_MODE=args.force_mode,
        IS_CAUSAL=args.causal,
        PV_USE_BF16=v.dtype is torch.bfloat16,
        HAS_STATS=True,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        grid=(q_blocks, batch, heads),
    )
    return _write_asm(compiled.asm, out_dir)


def _write_asm(asm: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, body in asm.items():
        suffix = {
            "ttir": ".ttir",
            "ttgir": ".ttgir",
            "llir": ".llir",
            "ptx": ".ptx",
            "cubin": ".cubin",
        }.get(name, f".{name}")
        path = out_dir / f"gate1{name if suffix.startswith('.') else ''}{suffix}"
        is_binary = isinstance(body, (bytes, bytearray))
        mode = "wb" if is_binary else "w"
        with path.open(mode) as handle:
            handle.write(body if is_binary else str(body))
        written[name] = str(path)
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["generic", "mass_specialized"], default="generic")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-q", type=int, default=1024)
    parser.add_argument("--seq-k", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--tile-size-q", type=int, default=64)
    parser.add_argument("--active-blocks", type=int, default=1)
    parser.add_argument("--peak", type=float, default=8.0)
    parser.add_argument("--force-mode", type=int, default=0)
    parser.add_argument("--skip-predicate", choices=["mass", "value_bound"], default="mass")
    parser.add_argument("--error-budget", type=float, default=1e-3)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--out-dir", default="artifacts/gate1_ir")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    q, k, v = _make_tensors(args)
    out_dir = Path(args.out_dir) / args.kernel / f"mode_{args.force_mode}"
    if args.kernel == "mass_specialized":
        written = _dump_mass(args, q, k, v, out_dir)
    else:
        written = _dump_generic(args, q, k, v, out_dir)
    print(json.dumps({"written": written}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
