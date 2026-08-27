# Unified Prefill And Training Attention API

## Result

StreamAttn now exposes exact prefill and differentiable attention through the
same semantic planner used by decode:

```python
import stream_attention as stream_attn

prefill_output = stream_attn.prefill(q, k, v)
training_output = stream_attn.train(q, k, v)
```

Both calls lower through:

```text
AttentionProblem -> AttentionTilePlan -> AttentionBackendPlan
```

The initial contract accepts `[B, S, H, D]` tensors, schedules every KV tile,
and carries the exact semantic guarantee. Supported CUDA calls execute the
Triton online-softmax forward/backward; other devices or unsupported training
features use the exact PyTorch SDPA fallback.

## H100 Correctness Gate

The public calls were run on one NVIDIA H100 80GB HBM3 with PyTorch 2.7.1 and
CUDA 12.8 at `B1, S128, H4, D64`, FP32, causal attention.

| Measurement | Result |
|---|---:|
| Prefill backend | `triton_online_softmax` |
| Training backend | `triton_online_softmax_autograd` |
| Prefill max absolute error vs SDPA | `0.0021913` |
| `dQ` max absolute error vs SDPA | `0.0050246` |
| `dK` max absolute error vs SDPA | `0.0049886` |
| `dV` max absolute error vs SDPA | `0.0045555` |

A second run at the same shape in BF16 also selected both native backends. Its
prefill max absolute error was `0.0078125`; `dQ`, `dK`, and `dV` were each
within `0.015625` max absolute error of SDPA.

A broader H100 regression then passed `10/10` CUDA tests covering the public
FP32/BF16 calls, non-tile-aligned masks, deterministic dropout, ALiBi, and
forward/backward parity. That gate also caught and fixed padded columns in a
partial final KV tile contributing to the non-causal softmax denominator.

The gate exposed and fixed three dormant backward-path defects: duplicate
Triton kernel symbol shadowing, BSHD/BHSD stride inversion, and atomic gradient
corruption caused by autotuning without resetting accumulation buffers, and
missing final-tile masking in non-causal attention. It also replaced a
tile-local softmax Jacobian reduction with the exact row-global identity:

```text
D_i = sum_j(P_ij * dP_ij) = dot(dO_i, O_i)
dS_ij = P_ij * (dP_ij - D_i)
```

## Native Compact GQA

The same public API now accepts `Hq > Hkv` when `Hq` is divisible by `Hkv`.
Native CUDA execution keeps K/V compact. A query head `h` reads:

```text
kv_head(h) = floor(h / (Hq / Hkv))
```

The mapping is performed inside both streaming kernels. Forward reads compact
K/V directly; backward accumulates the grouped query-head contributions into
compact dK/dV. No repeated K/V tensor is materialized on the native path. The
fallback labels physical expansion explicitly as `torch_sdpa_gqa_expanded`.

Correctness gates passed on both NVIDIA H100 and B200:

| GPU | Shapes | Dtypes | Result |
|---|---|---|---:|
| H100 / SM90 | B1, S128, Hq8/Hkv2, D64/D128 | FP32, BF16 | `4/4` native GQA cases passed forward/backward |
| B200 / SM100 | B1, S128, Hq8/Hkv2, D64/D128 | FP32, BF16 | `4/4` native GQA cases passed forward/backward |

The B200 gate exposed architecture-dependent resource limits that an H100-only
test would miss. D64 backward uses a smaller query tile on SM100, and D128 uses
`32x32` backward tiles plus a `32x64` forward tile on SM100. H100 retains the
larger valid D128 forward tile. These are resource-valid schedules, not a claim
that the first lowering is performance-optimal.

## H100/B200 Performance Phase

`benchmarks/profile_functional_gqa_phase.py` compares complete native forward
and forward-plus-backward calls with forced PyTorch Flash SDPA GQA when that
backend is available. BF16 measurements covered B1/B2 and S128/S512/S1024.

| GPU / shape | Prefill speedup vs Flash SDPA | Train speedup vs Flash SDPA |
|---|---:|---:|
| H100, Hq16/Hkv4, D64 | `0.31x-0.37x` | `0.29x-0.49x` |
| B200, Hq16/Hkv4, D64 | `0.21x-0.28x` | `0.26x-0.60x` |
| H100, Hq16/Hkv2, D128 | `0.15x-0.52x` | `0.13x-0.60x` |
| B200, Hq16/Hkv2, D128 | `0.19x-0.50x` | `0.09x-0.52x` |

All reported speedups are below `1.0x`: the compact native path is currently
slower than FlashAttention-class SDPA. Maximum BF16 output error in these
profiles was `0.015625`.

The result identifies the dataflow bottleneck rather than merely a tuning
constant. The current grid is Q-head-owned. For GQA group size `G`, that means:

```text
forward compact storage: yes
forward K/V traffic:     reloaded approximately G times
backward dK/dV:          G query-head contributions merged with atomics
```

The next performance backend must be KV-group-owned: load each K/V tile once,
process its grouped query heads cooperatively, and separate dQ production from
grouped dK/dV reduction. H100 should use a WGMMA-oriented schedule; B200 should
be designed around `tcgen05` MMA and TMEM rather than inheriting Hopper tiles.

## Boundary

This is an API, planner, compact-GQA, and cross-architecture correctness
milestone. It is not a prefill/training performance promotion. The measured
phase diagram rejects the current Q-head-owned GQA lowering as a competitive
FlashAttention replacement and gives the next backend a precise acceptance
condition: grouped K/V reuse must overcome launch, reduction, and
synchronization costs on both H100 and B200.
