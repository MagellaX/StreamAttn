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

## Backend Promotion (2026-08-14)

The measured CUDA/C++ source now lives under
`stream_attention/backends/sm90/` and is called by both the benchmark and the
serving exact-native runner. `ExactDecodePlan` compiles and allocates once,
prebinds query/output views and launch callables, and reuses 532,480 bytes of
FP32 split-state workspace. Its `run()` path performs only the producer and
merge launches.

The promoted serving specialization requires native head-major contiguous KV:

```text
Q: [B,1,Hq,D]
K/V: [B,Hkv,N,D]
```

It does not hide a BNHD-to-BHND cache transpose. Unsupported layouts and shapes
retain the existing StreamAttn exact fallback.

The final H100 promotion gate used log2 split-state LSE so partial and merge
rescaling remain aligned with the mainloop's `exp2` online softmax:

| Measurement | StreamAttn plan | FlashInfer | Speedup |
| --- | ---: | ---: | ---: |
| Independent median | 33.239 us | 33.468 us | 1.007x |
| Alternating paired median | 32.780 us | 33.116 us | 1.011x |
| Weakest paired trial | 32.962 us | 33.103 us | 1.004x |

All nine final paired trials won. The serving dispatcher selected
`sm90_transposed_gqa_wgmma_exact`, output remained deterministic and finite,
and max error versus the FP32 reference remained `9.01e-5`.

The performance margin remains narrow and environment-sensitive: an earlier
strict promotion rerun reached a `0.991x` paired median before the final log2
variant passed. Therefore the backend is promoted for exact ownership and this
guarded cell, while broader speed claims remain performance-gated by paired
measurements.

## Host-dispatch decomposition

An August 18 follow-up added a single C++ extension entry point that validates
the fixed buffers once and launches the existing producer and merge kernels on
the same CUDA stream. The original two-extension-call plan remained unchanged
as the control. Nine alternating H100 trials found:

| Comparison | Median ratio | Minimum | Wins |
| --- | ---: | ---: | ---: |
| Combined dispatch vs two-call plan | `0.99997x` | `0.99882x` | `4/9` |
| Combined dispatch vs FlashInfer | `0.98935x` | `0.98725x` | `0/9` |
| Two-call plan vs FlashInfer | `0.98900x` | `0.98816x` | `0/9` |

The combined path remained finite, deterministic, and retained the same
`9.01e-5` maximum FP32-reference error. It did not improve latency. This
disproves the interpretation that the earlier difference between independently
timed raw and planned paths was a removable `~0.55 us` Python dispatch tax.

The same run measured:

```text
producer:  28.128 us
merge:      3.324 us
raw total: 32.024 us
```

Independent serving timing reached `31.928 us` while paired measurements later
in the same process were near `32.05 us`, further demonstrating that independent
benchmark phases cannot be subtracted to attribute sub-microsecond overhead.
The combined entry point remains an experimental measurement path; the promoted
plan continues to use the established two-call implementation. The next exact
backend work is merge-kernel profiling and controlled merge variants, followed
by repeatability and adjacent-shape mapping.

Artifact:

```text
artifacts/gate0/transposed_wgmma_exact_combined_dispatch_modal_h100_20260818.json
```

## Scope

This is an exact-kernel victory for one important decode cell, not a universal
claim. The production specialization is narrowly guarded to H100, BF16, D64,
G8, B4, and head-major contiguous 32K KV.

It does establish two separate StreamAttn advantages:

```text
exact_native:     full exact attention can beat FlashInfer at a native cell
seed_only_native: validated cells can avoid most KV work for larger gains
```

The next engineering gate is to map adjacent shapes (`B2/B8`, `16K/64K`, other
GQA groups, and D128) without weakening the exact output gate.

## Artifact

The complete raw result is stored at:

```text
artifacts/gate0/transposed_wgmma_exact_native_gate_h100_20260813.json
artifacts/gate0/transposed_wgmma_exact_backend_promoted_modal_h100_20260814.json
```
