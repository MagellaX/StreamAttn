# StreamAttention: Fused Online Softmax Attention

A high-performance attention mechanism that computes softmax normalization in a single streaming pass using running accumulators (online softmax). The design achieves O(N) memory, strong numerical stability via log-sum-exp, and competitive throughput with modern flash attention baselines.

This repository provides:
- A production-oriented `StreamAttention` module for drop-in use in transformer models
- A fused online softmax kernel in Triton with safe PyTorch SDPA fallbacks
- A FlashAttention-3 baseline wrapper
- Benchmark and accuracy CLIs for reproducible comparisons
- Utilities for memory profiling and KV-cache compression
- Optional integration helpers for Hugging Face models


## Installation

The project depends on PyTorch (CUDA optional) and Triton (optional for the custom fused kernel).

```bash
# Stable release (from PyPI)
pip install stream-attention

# Optional extras
pip install "stream-attention[hf]"  # Hugging Face integration helpers

# Development install (from a cloned repo)
pip install -e .
```

Notes:
- If the environment is system-managed, you may need to pass `--break-system-packages` to pip or use a virtual environment.
- Triton is optional. If Triton is not available, the implementation falls back to PyTorch SDPA (flash backend on CUDA where available).


## Quick Start

```python
import torch
from stream_attention import StreamAttention, StreamAttentionConfig

# Configure the attention
config = StreamAttentionConfig(
    num_heads=32,
    head_dim=128,
    tile_size_q=128,
    tile_size_k=64,
    use_qkv_projections=True
)

# Create the module
attention = StreamAttention(config).cuda() if torch.cuda.is_available() else StreamAttention(config)

# Hidden-states path (with internal Q,K,V projections)
batch_size, seq_len = 2, 1024
hidden_dim = config.num_heads * config.head_dim
x = torch.randn(batch_size, seq_len, hidden_dim, device=attention.attention.device, dtype=(torch.float16 if torch.cuda.is_available() else torch.float32))

with torch.no_grad():
    y = attention(x)
print(y.shape)

# Explicit Q,K,V path
q = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim, device=x.device, dtype=x.dtype)
k = torch.randn_like(q)
v = torch.randn_like(q)
with torch.no_grad():
    y_qkv = attention(q, k, v, causal=True)
print(y_qkv.shape)
```


## API Reference

### StreamAttention
- Purpose: High-level module. Accepts either `hidden_states` ([B, T, H*D]) or explicit `(query, key, value)` ([B, T, H, D]).
- Signature (selected):
  - `forward(hidden_states: Tensor, ..., use_cache: bool=False, causal: bool=True)` → `Tensor` or `(Tensor, (k, v))` if `use_cache=True`
  - `forward(query: Tensor, key: Tensor, value: Tensor, causal: bool=True)` → `Tensor`
- Shapes: `[batch, seq, heads, dim]` for QKV mode.
- Dtypes: fp16/bf16 (CUDA), fp32 (CPU by default). On CPU, inputs upcast to fp32 if required.

### FusedOnlineAttention
- Purpose: Low-level fused online softmax attention (Triton when available; SDPA fallback otherwise).
- Signature (selected):
  - `forward(query, key, value, causal: bool=True, return_lse: bool=False)` → `Tensor` (and `lse` if requested)
  - `benchmark(seq_len: int, batch_size: int=1, warmup: int=10, iterations: int=100)` → metrics dict
- Autograd: If gradients are required, the module automatically falls back to PyTorch SDPA to ensure correct backward support. The Triton path is intended for forward-critical inference/benchmarking.

### FlashAttentionV3
- Purpose: Baseline using PyTorch SDPA with the flash backend on CUDA, falling back gracefully on CPU.
- Signature (selected):
  - `forward(query, key, value, causal: bool=True)` → `Tensor`
  - `benchmark(...)` → metrics dict

### StreamAttentionConfig
Selected fields (see source for full set):
- `num_heads`, `head_dim`
- `tile_size_q`, `tile_size_k`
- `use_fp16`, `use_qkv_projections`, `qkv_bias`, `use_layer_norm`, `dropout`
- `enable_distributed`
- `max_sequence_length`
- Ring/Star attention parameters for long-context variants
- `.from_env()` reads `STREAM_ATTENTION_*` variables; see `.env.example`


## Benchmarking vs FlashAttention-3

Two CLIs are provided:

```bash
# Performance comparison across sequence lengths
stream-attention-benchmark --seq 512 1024 2048 4096 --batch 1 --heads 8 --dim 64 --warmup 10 --iters 50

# Accuracy sanity check on random inputs
stream-attention-test --seq 1024 --batch 2 --heads 8 --dim 64 --dtype fp16
```

Behavior and methodology:
- On CUDA, the baseline uses PyTorch SDPA with the flash backend (FlashAttention-3 path). On CPU, both implementations use SDPA in fp32.
- Metrics reported: per-iteration latency, estimated TFLOPS, and approximate bandwidth based on tensor traffic. Measurements are averaged after warmup.
- The fused kernel enables Triton only when available, running on CUDA, and autograd is not required. Otherwise, SDPA is used to ensure correctness and training support.
- For reproducibility, fix random seeds, pin CUDA clocks if applicable, and isolate runs. Actual performance depends on GPU architecture, drivers, and PyTorch/Triton versions.

Example output (format):
```
SeqLen  Fused(ms)  Fused(TF)  FA3(ms)  FA3(TF)  FA3/Fused(ms)
  1024      0.650     90.12     0.700     83.70          1.08
```


## Kernel Design (Fused Online Softmax)

The fused kernel streams over K/V tiles while maintaining per-query running statistics for online softmax, avoiding materialization of the full attention matrix.

- Tiling:
  - Queries are processed in blocks of `TILE_M` (configurable via autotune); keys/values are streamed in tiles of `TILE_N`.
  - Head dimension `D` is processed vectorized. The kernel uses `tl.dot(q, tl.trans(k))` to compute `QK^T` for each tile.

- Online softmax with log-sum-exp:
  - Maintain `running_max[m]`, `acc_num[m, :]`, `acc_den[m]` for each query row `m`.
  - For each K/V tile:
    - `tile_max = max_j qk[m, j]`, `new_max = max(running_max[m], tile_max)`
    - Rescale previous accumulators by `exp(running_max - new_max)`
    - `exp_qk = exp(qk - new_max)`
    - `acc_num += exp_qk @ V_tile`, `acc_den += sum(exp_qk, axis=tile)`
    - `running_max = new_max`
  - Final output: `acc_num / acc_den[:, None]`. Log-sum-exp `lse = running_max + log(acc_den)` may be returned for analysis.

- Autotuning:
  - Multiple Triton configurations for `TILE_M`/`TILE_N`, warps, and stages are provided.
  - Kernel grid: `(ceil_div(M, TILE_M), batch, heads)`

- Numerical stability:
  - The log-sum-exp re-centering preserves stability across tiles and long sequences.
  - On CPU, fp16/bf16 inputs are upcast to fp32 where necessary.

- Backward:
  - The Triton path is presently forward-oriented. For training, the module selects PyTorch SDPA, which supports autograd.


## Distributed and Long-Context Variants

- Ring Attention (`stream_attention/core/ring_attention.py`): partitions sequences across devices with overlapped communication and computation. Suitable for near-infinite contexts with linear memory scaling.
- Star Attention (`stream_attention/core/star_attention.py`): two-phase approach (context encoding, then query processing with aggregated attention). Reduces end-to-end latency for long sequences while preserving accuracy.

Both modules follow numerically stable aggregation strategies (log-sum-exp weighted combining) and can be paired with KV compression.


## Memory Optimization Utilities

- KV-cache compression strategies: importance-based, chunk-based, quantized, and hybrid.
- Gradient checkpointing helpers for sequential modules.
- `MemoryProfiler` to snapshot and report peak allocations.

Example:
```python
from stream_attention.utils.memory import create_kv_compressor
comp = create_kv_compressor('hybrid')
ck, cv, stats = comp.compress(keys, values, compression_ratio=8.0)
print(stats)
```


## Hugging Face Integration

Helpers are provided to replace attention modules in HF models:

```python
from transformers import AutoModel
from stream_attention.integration.hf import replace_llama_attention
from stream_attention import StreamAttentionConfig

model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")
cfg = StreamAttentionConfig(num_heads=model.config.num_attention_heads, head_dim=model.config.hidden_size // model.config.num_attention_heads)
num_replaced = replace_llama_attention(model, cfg)
print(f"Replaced {num_replaced} attention modules")
```


## Supported Environments

- PyTorch 2.0+
- CUDA: fp16 and bf16 supported via SDPA (flash backend where available). Triton kernel requires CUDA.
- CPU: falls back to SDPA with fp32 compute for correctness and stability.
- Distributed: query sharding is supported in the fused module for multi-GPU; ring/star provide long-context strategies.


## Roadmap

- Backward implementation for the Triton fused kernel
- Advanced pipelining (warp specialization, asynchronous staging) and Hopper-specific paths (WGMMA/TMA)
- Full support for attention masks and dropout in the fused kernel
- Extended autotune coverage across architectures and sequence regimes
- Optional FP8 path with block-wise scaling


## Development and Testing

- Benchmarks: `stream-attention-benchmark` CLI
- Accuracy checks: `stream-attention-test` CLI
- Examples: `examples/` directory includes basic usage, integrations, and long-context runs
- Environment variables: see `.env.example`; `StreamAttentionConfig.from_env()` can bootstrap configuration


## License

MIT License. See `LICENSE` for details.
