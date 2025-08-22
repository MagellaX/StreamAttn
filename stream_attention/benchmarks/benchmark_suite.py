import argparse
import json
import torch
from typing import List, Dict

from stream_attention.core.fused_online_attention import FusedOnlineAttention
from stream_attention.core.flashattention_v3 import FlashAttentionV3
from stream_attention.core.config import StreamAttentionConfig


def _benchmark_module(module, seq_len, batch_size, warmup, iterations):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        nh = getattr(module, "num_heads")
        hd = getattr(module, "head_dim")
        q = torch.randn(batch_size, seq_len, nh, hd, device=device, dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for _ in range(warmup):
                module(q, k, v, causal=True)
        if device.type == "cuda":
                torch.cuda.synchronize()
        import time
        start = time.time()
        for _ in range(iterations):
                module(q, k, v, causal=True)
        if device.type == "cuda":
                torch.cuda.synchronize()
        elapsed = (time.time() - start) / iterations
        flops = 4.0 * batch_size * nh * seq_len * seq_len * hd
        tflops = flops / elapsed / 1e12
        bytes_per_el = torch.tensor([], dtype=dtype).element_size()
        memory_bytes = 3 * batch_size * seq_len * nh * hd * bytes_per_el
        bandwidth = memory_bytes / elapsed / 1e9
        return {"time_ms": elapsed * 1000.0, "tflops": tflops, "bandwidth_gb_s": bandwidth}


def run_bench(seq_lens: List[int], batch_size: int, num_heads: int, head_dim: int, warmup: int, iters: int) -> Dict[int, Dict[str, float]]:
        cfg = StreamAttentionConfig(num_heads=num_heads, head_dim=head_dim, use_fp16=torch.cuda.is_available())
        fused = FusedOnlineAttention(num_heads=num_heads, head_dim=head_dim,
                                     dtype=(torch.float16 if torch.cuda.is_available() else torch.float32))
        fa3 = FlashAttentionV3(cfg)
        results = {}
        for L in seq_lens:

                fr = fused.benchmark(seq_len=L, batch_size=batch_size, warmup=warmup, iterations=iters)


                fr = fused.benchmark(seq_len=L, batch_size=batch_size, warmup=warmup, iterations=iters)

                fr = _benchmark_module(fused, L, batch_size, warmup, iters)

                ar = fa3.benchmark(seq_len=L, batch_size=batch_size, warmup=warmup, iterations=iters)
                results[L] = {
                        "fused_time_ms": fr["time_ms"],
                        "fused_tflops": fr["tflops"],
                        "fa3_time_ms": ar["time_ms"],
                        "fa3_tflops": ar["tflops"],
                        "speedup_vs_fa3": ar["time_ms"] / fr["time_ms"] if fr["time_ms"] > 0 else float("inf"),
                }
        return results


def main():
	parser = argparse.ArgumentParser(description="StreamAttention vs FlashAttention-3 Benchmark")
	parser.add_argument("--seq", nargs="*", type=int, default=[512, 1024, 2048, 4096])
	parser.add_argument("--batch", type=int, default=1)
	parser.add_argument("--heads", type=int, default=8)
	parser.add_argument("--dim", type=int, default=64)
	parser.add_argument("--warmup", type=int, default=10)
	parser.add_argument("--iters", type=int, default=50)
	parser.add_argument("--json_out", type=str, default="")
	args = parser.parse_args()
	res = run_bench(args.seq, args.batch, args.heads, args.dim, args.warmup, args.iters)
	print("SeqLen\tFused(ms)\tFused(TF)\tFA3(ms)\tFA3(TF)\tFA3/Fused(ms)")
	for L in args.seq:
		r = res[L]
		print(f"{L}\t{r['fused_time_ms']:.3f}\t{r['fused_tflops']:.2f}\t{r['fa3_time_ms']:.3f}\t{r['fa3_tflops']:.2f}\t{r['speedup_vs_fa3']:.2f}")
	if args.json_out:
		with open(args.json_out, "w") as f:
			json.dump(res, f, indent=2)


if __name__ == "__main__":
	main()