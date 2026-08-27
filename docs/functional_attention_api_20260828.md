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

## Boundary

This is an API, planner, and correctness milestone, not a prefill performance
claim. Native prefill/training currently requires equal Q and KV head counts.
GQA is rejected instead of materializing duplicated KV heads. The next native
expansion is direct GQA lowering in the fused forward/backward kernels followed
by H100/B200 phase diagrams against FlashAttention-class baselines.
