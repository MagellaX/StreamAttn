# Exact-Native H100 B4 Gate

Date: 2026-08-13

## Result

StreamAttn passed its first native exact-attention performance gate against a
matching FlashInfer batch-decode path.

```text
GPU:       NVIDIA H100 80GB HBM3
dtype:     BF16
B:         4
Hq/Hkv:    16/2 (GQA group size 8)
D:         64
KV length: 32768
mode:      exact attention
splits:    32
```

The promotion run used 30 warmups, 500 timed iterations, nine repeated timing
samples, and nine alternating paired comparisons.

| Measurement | StreamAttn | FlashInfer | Speedup |
| --- | ---: | ---: | ---: |
| Independent median | 32.300 us | 33.420 us | 1.035x |
| Paired median | 32.421 us | 33.060 us | 1.020x |
| Weakest paired trial | 32.539 us | 33.060 us | 1.016x |

Every paired trial was faster than FlashInfer.

## Numerical Gate

```text
deterministic repeat delta:       0
non-finite outputs:               0
max error vs FP32 dense reference: 9.01e-5
max difference vs FlashInfer BF16: 1.22e-4
mean difference vs FlashInfer:     2.04e-5
```

The result is exact at the attention-algorithm level: both paths evaluate all
32K KV tokens and use numerically stable online softmax. Differences are normal
low-precision kernel-order effects, not token selection or sparsity.

## Architecture

The kernel transposes GQA decode into Hopper's useful tensor-core geometry:

```text
QK: K_tile [64,64] @ Q_group.T [64,8] -> scores [64,8]
PV: V_tile.T [64,64] @ P [64,8]       -> output [64,8]
```

Both operations use native `m64n8k16` WGMMA, so the GQA group occupies eight
real columns without padding it to 16. The pipeline includes:

1. 128-bit `cp.async` K/V loads.
2. Double-buffered shared-memory K/V tiles.
3. FP32 online-softmax max/sum state.
4. BF16 probability staging into a canonical Major-MN interleaved score tile.
5. FP32 PV accumulation.
6. Exact split-state output and an LSE-weighted merge kernel.

The best configuration used 32 context splits, or 256 producer CTAs. Fewer
splits under-filled the H100; more splits increased partial-state and merge
cost. This is the measured form of the scheduler objective:

```text
minimize T_partial(C) + T_merge(C)
subject to enough producer CTAs and exact output semantics
```

## Scope

This is an exact-kernel victory for one important decode cell, not a universal
claim. The current implementation is an embedded CUDA/CUTLASS benchmark and is
narrowly specialized to H100, BF16, D64, G8, B4, and contiguous 32K KV.

It does establish two separate StreamAttn advantages:

```text
exact_native:     full exact attention can beat FlashInfer at a native cell
seed_only_native: validated cells can avoid most KV work for larger gains
```

The next engineering gate is to extract this kernel into the native backend,
add plan/workspace reuse, and measure adjacent shapes (`B2/B8`, `16K/64K`,
other GQA groups, and D128) without weakening the exact output gate.

## Artifact

The complete raw result is stored at:

```text
artifacts/gate0/transposed_wgmma_exact_native_gate_h100_20260813.json
```
