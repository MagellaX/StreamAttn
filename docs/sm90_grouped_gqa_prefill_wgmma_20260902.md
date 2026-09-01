# SM90 Natural-Orientation Grouped GQA Prefill Canary

## Question

The grouped Triton floor established that compact GQA K/V reuse improves
StreamAttn's first exact prefill path but does not reach Flash SDPA. This
experiment tests whether changing the physical operator to native Hopper
WGMMA closes that gap.

The kernel keeps the exact causal attention equation and streaming online
softmax. It changes ownership from one CTA per query head to one CTA per KV
head and query-position tile. Query heads sharing that KV head therefore reuse
the same staged K/V tile.

For group size `G`, a CTA with `M` flattened query-head rows covers:

```text
query positions per CTA = M / G
```

The first canary used `M=64`. The second used two 64-row consumer warpgroups,
so `M=128`, while staging K/V once for both consumers.

## Implementation

The experimental backend is intentionally outside production dispatch:

```text
stream_attention/backends/sm90/grouped_gqa_prefill.py
streamattn_grouped_wgmma_prefill_kernel
```

Its measured specialization is:

```text
GPU: NVIDIA H100 80GB HBM3 (SM90)
dtype: BF16
layout: contiguous BSHD
head dimension: 128
query heads: 16
GQA groups: G4 and G8
batch: 1
sequence: 128, 256, 512, 1024, 2048, 4096
semantics: exact causal self-attention
baseline: forced Flash SDPA under fixed-address CUDA graph replay
```

Both variants use natural-orientation
`SM90_64x64x16_F32BF16BF16_SS` WGMMA for QK and PV. K/V moves through
double-buffered `cp.async` shared-memory stages. Each row maintains an FP32
running maximum, denominator, and output numerator. No attention matrix or
expanded GQA K/V tensor is materialized.

The two-consumer variant launches 256 threads. Consumer warpgroups own
separate Q, probability, softmax, and output state while sharing each K/V
stage. This halves the CTA count relative to the 64-row canary without changing
the exact QK/PV arithmetic.

## Correctness

All 12 H100 screen cells passed the strict canary checks:

```text
maximum output absolute error: 0.00390625
maximum output relative L2:     0.002014
maximum checked LSE error:      9.5367e-7
```

The LSE check uses an FP32 dense reference for sequences through 512 tokens.

## Performance

Speedup is `Flash SDPA time / StreamAttn time`, so values below one are losses.

| Group | Sequence | 64-row CTA | 128-row / two-consumer CTA |
|---:|---:|---:|---:|
| G4 | 128 | `0.352x` | `0.408x` |
| G4 | 256 | `0.436x` | `0.451x` |
| G4 | 512 | `0.579x` | `0.527x` |
| G4 | 1024 | `0.520x` | `0.544x` |
| G4 | 2048 | `0.560x` | `0.525x` |
| G4 | 4096 | `0.530x` | `0.528x` |
| G8 | 128 | `0.352x` | `0.420x` |
| G8 | 256 | `0.442x` | `0.459x` |
| G8 | 512 | `0.596x` | `0.537x` |
| G8 | 1024 | `0.526x` | `0.551x` |
| G8 | 2048 | `0.558x` | `0.527x` |
| G8 | 4096 | `0.547x` | `0.539x` |

Neither variant won a paired trial. The two-consumer resource change explains
why reuse did not translate into net throughput:

| Variant | Registers / thread | Dynamic shared | Active CTAs / SM |
|---|---:|---:|---:|
| 64-row / one consumer | 220 | 59,008 bytes | 2 |
| 128-row / two consumers | 223 | 85,120 bytes | 1 |

The second consumer nearly doubles the register footprint per CTA, increases
shared memory, and reduces residency to one CTA per SM. Both consumers also
cross CTA-wide synchronization points around score reduction, probability
materialization, V readiness, and K/V stage turnover. The result is a small
short-sequence gain but no long-sequence trend toward parity.

## Decision

Do not promote this backend and do not continue consumer-symmetric tile
tuning. The experiment rejects the hypothesis that natural WGMMA plus grouped
K/V reuse is sufficient.

A future H100 prefill attempt must change the pipeline, not just its tile:

```text
dedicated TMA producer warpgroup
asynchronous producer/consumer barriers
consumer register-budget control
less CTA-wide softmax synchronization
larger WGMMA work per synchronization epoch
```

That is a distinct FlashAttention-3-class schedule. It should begin with a
resource and instruction-overlap floor before another complete attention
kernel is built. Until such a floor approaches parity, the exact phase
compiler should keep these cells on the fastest correct external fallback.

## Artifacts

```text
artifacts/gate0/sm90_grouped_gqa_prefill_screen_modal_h100_20260902.json
artifacts/gate0/sm90_grouped_gqa_prefill_m64n64x2_screen_modal_sm90_20260902.json
```
