import argparse
import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from stream_attention.core.fused_online_attention import FusedOnlineAttention
from stream_attention.core.flashattention_v3 import FlashAttentionV3
from stream_attention.core.config import StreamAttentionConfig


def main():
    parser = argparse.ArgumentParser(
        description="Accuracy comparison: FusedOnlineAttention vs FlashAttentionV3"
    )
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument(
        "--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"]
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    if device.type != "cuda" and dtype != torch.float32:
        print("Non-CUDA device; forcing fp32")
        dtype = torch.float32

    cfg = StreamAttentionConfig(
        num_heads=args.heads, head_dim=args.dim, use_fp16=(dtype == torch.float16)
    )
    fused = FusedOnlineAttention(
        num_heads=args.heads, head_dim=args.dim, dtype=dtype
    ).to(device)
    fa3 = FlashAttentionV3(cfg).to(device)

    q = torch.randn(
        args.batch, args.seq, args.heads, args.dim, device=device, dtype=dtype
    )
    k = torch.randn(
        args.batch, args.seq, args.heads, args.dim, device=device, dtype=dtype
    )
    v = torch.randn(
        args.batch, args.seq, args.heads, args.dim, device=device, dtype=dtype
    )

    with torch.no_grad():
        o1 = fused(q, k, v, causal=True)
        o2 = fa3(q, k, v, causal=True)
        diff = (o1.float() - o2.float()).abs()
        print(
            f"Mean abs diff: {diff.mean().item():.6f}, Max abs diff: {diff.max().item():.6f}"
        )


if __name__ == "__main__":
    main()


