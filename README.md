# StreamAttention: Fused Online Softmax Attention

A high-performance attention mechanism that computes softmax normalization in a single streaming pass using running accumulators (online softmax). The design achieves O(N) memory, strong numerical stability via log-sum-exp, and competitive throughput with modern flash attention baselines.

This repository provides:
- A production-oriented `StreamAttention` module for drop-in use in transformer models
- A fused online softmax kernel in Triton with safe PyTorch SDPA fallbacks
- A FlashAttention-3 baseline wrapper
- Benchmark and accuracy CLIs for reproducible comparisons
- Utilities for memory profiling and KV-cache compression
- Optional integration helpers for Hugging Face models

## Recent Updates

- Added a single-sweep Triton backward path (streaming dQ/dK/dV using saved `lse`) that mirrors the forward pass, including masks, dropout, and ALiBi.
- Triton forward now supports boolean/additive masks, dropout, deterministic Philox seeding, and ALiBi bias without SDPA fallback.
- Expanded tests/docs covering mask/dropout/ALiBi parity plus deterministic mode usage.



## Installation

The project depends on PyTorch (CUDA optional) and Triton (optional for the custom fused kernel).

```bash
# Editable install (preferred for development)
pip install -e .

# Optional extras
pip install -e .[hf]        # Hugging Face integration helpers
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
attention = (StreamAttention(config).cuda() if torch.cuda.is_available() else StreamAttention(config))

# Hidden-states path (with internal Q,K,V projections)
batch_size, seq_len = 2, 1024
hidden_dim = config.num_heads * config.head_dim
x = torch.randn(batch_size, seq_len, hidden_dim, device=attention.attention.device, dtype=(torch.float16 if torch.cuda.is_available() else torch.float32))

with torch.no_grad():
    y = attention(hidden_states=x)
print(y.shape)

# Explicit Q,K,V path (supports attn_mask/dropout/ALiBi in Triton, falling back to SDPA when needed)
q = torch.randn(batch_size, seq_len, config.num_heads, config.head_dim, device=x.device, dtype=x.dtype)
k = torch.randn_like(q)
v = torch.randn_like(q)
with torch.no_grad():
    # Example boolean mask [B, S_k]
    key_padding_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
    key_padding_mask[:, -16:] = False  # mask out last 16 positions
    y_qkv = attention(q, k, v, causal=True, attention_mask=key_padding_mask)
print(y_qkv.shape)
```

## Seed-Only Serving Wedge

The first deployable StreamAttn decode route is intentionally narrow and
fail-closed: Qwen2.5-0.5B layer 8, post-RoPE true-GQA tensors, 32K KV bucket,
fp16, batch >= 4. It uses the packaged seed-only policy when the request
matches and otherwise stays inside StreamAttn exact-native mode. FlashInfer is
kept as a benchmark/reference injection, not a required serving fallback.

Validated seed-only cells are indexed by
`stream_attention/policies/registry.json`. Use
`list_packaged_gate0_seed_only_batched_policies()` or
`find_packaged_gate0_seed_only_batched_policies(...)` to discover green routes
before loading an artifact. Today the registry contains six green Qwen2.5-0.5B
32K cells: L1, L2, L5, L6, L8, and L18. New frontier-model or
additional-layer routes should enter through this registry only after their
runtime and distribution-safety artifacts pass.

```python
from stream_attention import StreamAttnSeedOnlyDecodeService

service = StreamAttnSeedOnlyDecodeService.from_packaged()

out, info = service.run(q, k_cache, v_cache, mode="auto")
print(info.to_dict())
```

CPU-safe fail-closed smoke:

```bash
python examples/seed_only_serving_decode.py --batch 4 --kv-len 128 --dtype fp32
```

Real serving-shape smoke with FlashInfer as an external reference backend:

```bash
python examples/seed_only_serving_decode.py --backend flashinfer --device cuda --batch 4 --kv-len 32768 --dtype fp16
```

## Engine Direction

The narrow Qwen L8 route is the first executable wedge, not the final goal.
StreamAttn is being shaped as a self-owned attention serving engine:

- `streamattn_exact_native`: exact decode owned by StreamAttn. It starts as the
  internal dense reference path and is the replacement point for TK/CUDA exact
  kernels.
- `gate0_seed_only_batched`: the first optimized native mode, enabled only for
  validated logit-safe cells.
- FlashInfer and FlashAttention-class kernels are development baselines and
  reference backends, not required serving fallbacks.
- General frontier-model coverage requires a policy compiler over model,
  layer, KV-length, batch, dtype, GQA/MQA/MHA shape, and device buckets.

The seed-only backend is guided by explicit kernel economics. For true GQA with
group size `G = Hq / Hkv`, StreamAttn can intentionally duplicate tiny seed K/V
reads across Q heads when:

```text
G * seed_tokens / kv_len << 1
```

For the packaged Qwen L8 32K route this is about `8.2%`, while the seed window
itself is only `384 / 32768 = 1.17%` of the KV cache. The autotuner in
`benchmarks/profile_seed_kernel_mode_autotune.py` reports these ratios, CTA
counts, and whether the next kernel should use head-private direct seed,
head-private split-seed, or a GQA-shared seed path.

The first two-kernel split-seed prototype is intentionally diagnostic: it writes
partial online-softmax states and merges them in a second kernel. The H100
below-B8 artifact in `artifacts/seed_only_split_seed_below8_h100.json` shows
this path is numerically correct against direct seed-only but not product
profitable. On that run, direct seed already wins at B4, while split-seed loses
to direct seed for every batch and does not recover B1/B2. Merge and extra
launch overhead dominate, so batch `<4` needs a lower-overhead single-kernel
CUDA/TK cooperative split design, not the current two-kernel Triton
decomposition.

The direct seed-only batch-threshold gate in
`artifacts/seed_only_direct_below8_h100_gate.json` narrows that target further.
On captured Qwen L8 32K rows, the raw direct seed kernel first beats the
FlashInfer exact reference at batch 4, and the planned direct service path now
clears the same product gate at batch 4. The ordinary wrapper route remains as
a comparison path, but `StreamAttnSeedOnlyDecodeService` defaults to a bound
planned-direct runner for eligible fixed-buffer CUDA decode loops. The measured
native route rule is therefore:

```text
B >= 4: head_private_direct_seed
B < 4: exact_native until a single-kernel cooperative split-seed path proves out
```

The next expansion is not new sparse-attention policy work; it is extending
that planned/native path across more layers, lengths, devices, and model
families, while treating B1/B2 as a separate backend-kernel project.

The artifact-driven policy compiler turns sweep/rollout evidence into policy
cells and registry metadata:

```bash
python benchmarks/compile_streamattn_seed_policy.py \
  --sweep-json artifacts/gate0/seed_only_layer_sweep_l0_l23_b4_h100.json \
  --closed-loop-dir artifacts/gate0 \
  --summary-json artifacts/gate0/compiled_seed_policy_qwen25_05b_32k_b4_h100.json
```

For the current Qwen2.5-0.5B 32K B4 evidence, the compiler reproduces the six
green layers L1, L2, L5, L6, L8, and L18. It requires both the last-32 layer
sweep gate and per-layer 32-step teacher-forced, greedy, and coupled top-p
rollout gates before a cell can enter the registry.

For new model buckets, use the Modal policy pipeline to run sweep, validate
candidates, and compile a summary in one H100 job:

```bash
modal run benchmarks/modal_compile_streamattn_seed_policy.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --layers 0-27 \
  --kv-len 32768 \
  --batch-size 4 \
  --output-dir artifacts/gate0/qwen25_15b_32k_b4
```

The pipeline only runs closed-loop rollout for sweep-passing candidate layers.
It writes the sweep, per-candidate rollout artifacts, compiled policy summary,
and `seed_policy_pipeline_summary.json` into the output directory.

`StreamAttnSeedOnlyDecodeService.plan_direct_seed_only(...)` is the first step
in that direction: it validates policy and tensor invariants once, binds fixed
Q/K/V/output buffers, and returns a `StreamAttnSeedOnlyDirectRunner` whose
steady-state `run()` path launches the prechecked direct seed kernel without
per-step routing.

The B4 promotion is backed by `artifacts/seed_only_direct_below8_h100_gate.json`
for runtime and `artifacts/gate0/seed_only_b4_closed_loop_h100.json` for
32-step teacher-forced, greedy, coupled top-p, and forced-same-token safety.


## API Reference

### StreamAttention
- Purpose: High-level module. Accepts either `hidden_states` ([B, T, H*D]) or explicit `(query, key, value)` ([B, T, H, D]).
- Signature (selected):
  - `forward(query, key, value, causal: bool=True, return_lse: bool=False, attention_mask: Optional[Tensor]=None, dropout_p: float=0.0, alibi_slopes: Optional[Tensor]=None, deterministic: Optional[bool]=None)` -> `Tensor` (and `lse` if requested)
  - `benchmark(seq_len: int, batch_size: int=1, warmup: int=10, iterations: int=100)` -> metrics dict
  - `set_deterministic(enabled: bool, seed: Optional[int]=None)` -> control deterministic dropout/mask behavior
- Autograd: The Triton kernel now runs a single-sweep backward pass (streaming dQ/dK/dV using the saved `lse`) covering masks, dropout, and ALiBi. PyTorch SDPA is used only when Triton is unavailable.

### Multihead-style wrapper
Use `create_stream_attention` to obtain an attention layer with a familiar
`nn.MultiheadAttention` interface. Triton kernels are used automatically when
available, otherwise PyTorch's SDPA backend is selected:

```python
import torch
from stream_attention import create_stream_attention

mha = create_stream_attention(embed_dim=512, num_heads=8, batch_first=True)
x = torch.randn(2, 16, 512)
out, _ = mha(x, x, x)
```

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
- The fused kernel uses Triton on CUDA for forward and single-sweep backward when the custom path is available. Mask, dropout, ALiBi, and deterministic replay are covered by the Triton path; SDPA remains the safety fallback when Triton or a requested shape/backend is unavailable.
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
  - The Triton autograd path recomputes QK tiles using saved `lse`, streams dQ/dK/dV, and avoids materializing the attention matrix. PyTorch SDPA remains the fallback for unsupported environments.


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
- Distributed: query sharding is supported in the fused module for multi-GPU; ring/star provides long-context strategies.


## Roadmap

- Native RoPE / relative positional bias fusion in the Triton kernel (forward + backward)
- Advanced pipelining (warp specialization, asynchronous staging) and Hopper-specific paths (WGMMA/TMA)
- Extended autotune coverage across architectures and sequence regimes
- Optional FP8 path with block-wise scaling


## Development and Testing

- Benchmarks: `stream-attention-benchmark` CLI
- Accuracy checks: `stream-attention-test` CLI
- Examples: `examples/` directory includes basic usage, integrations, and long-context runs
- Environment variables: see `.env.example`; `StreamAttentionConfig.from_env()` can bootstrap configuration


## License

Apache License. See `LICENSE` for details.
