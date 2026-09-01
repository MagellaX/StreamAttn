# Exact-Native A100 D64 Phase Diagram

Date: 2026-08-31

## Result

StreamAttn now has a guarded exact BF16 decode backend for NVIDIA A100 40GB.
It computes attention over every KV token. The promoted kernel uses:

- direct unpadded `[B,Hq,D]` query loads into G4/G8 tensor-core tiles;
- head-major contiguous `[B,Hkv,N,D]` K/V;
- double-buffered `cp.async` K/V staging;
- exact online softmax in each producer warp;
- four-way exact state reduction inside each CTA; and
- an active-head-only final log-sum-exp merge into `[B,Hq,D]`.

The corrected fused matrix beat FlashInfer 0.6.13 on all eight measured cells:

| Cell | StreamAttn | FlashInfer | Speedup | Logical chunks |
|---|---:|---:|---:|---:|
| B1 / G8 / 32K | `22.60 us` | `33.37 us` | `1.477x` | 128 |
| B2 / G8 / 32K | `35.33 us` | `41.24 us` | `1.167x` | 128 |
| B4 / G8 / 32K | `63.70 us` | `71.89 us` | `1.129x` | 128 |
| B8 / G8 / 32K | `109.86 us` | `117.23 us` | `1.067x` | 64 |
| B4 / G4 / 32K | `109.53 us` | `117.17 us` | `1.070x` | 64 |
| B4 / G8 / 16K | `32.39 us` | `41.05 us` | `1.267x` | 128 |
| B4 / G8 / 64K | `110.86 us` | `118.72 us` | `1.071x` | 128 |
| B4 / G8 / 32K, seed 7 | `63.46 us` | `71.54 us` | `1.127x` | 128 |

Aggregate: `8/8` wins, `1.067x` worst, `1.128x` median, and `1.477x` best.

## Baseline correction

The first matrix inherited a single-row FlashInfer helper. It indexed batch row
zero, so its B>1 timing and quality comparisons were invalid. That artifact was
not used for promotion.

The corrected benchmark plans FlashInfer's
`BatchDecodeWithPagedKVCacheWrapper` once, uses page-16 NHD KV, tensor cores,
and the `auto` backend, then times only `wrapper.run`. All FlashInfer outputs in
the corrected matrix were compared with the same FP32 dense true-GQA reference.

Across the fused matrix, StreamAttn and FlashInfer had the same per-cell maximum
BF16 error scale (`1.22e-4` or `2.44e-4`). StreamAttn mean absolute error ranged
from `1.57e-5` to `2.87e-5`.

## Why boundary fusion was required

The grouped computational core first won all eight corrected cells, but its
benchmark input was already padded to `[B,Hkv,16,D]`. A conservative serving
wrapper using ordinary PyTorch copies measured:

```text
Q pack + grouped exact core + active-row extraction
```

Those copies added `8.6-34.2 us` and produced `0/8` net wins. This ruled out a
generic layout-wrapper integration.

The fused kernel instead maps standard Q directly into the register tile with
masked zero rows and writes only active heads from the final merge. The result
returned to `8/8` wins and was slightly faster than the padded core in several
cells because padded output rows are never merged.

## Allocation-free serving canary

The production extension exposes an `_out` entry point. `ExactDecodePlan`
preallocates BF16 partial outputs, FP32 LSE states, and the final output once.
A fresh B4/G8/32K seed measured:

```text
StreamAttn preallocated plan: 63.16 us
FlashInfer batched paged:     73.02 us
speedup:                       1.156x
max error for both:            1.22e-4
```

## Promoted cells

The runtime uses a discrete phase table:

```text
G8:
  B1 / 32K -> C128
  B2 / 32K -> C128
  B4 / 16K -> C128
  B4 / 32K -> C128
  B4 / 64K -> C128
  B8 / 32K -> C64

G4:
  B4 / 32K -> C64
```

Only A100, BF16, D64, contiguous head-major KV, G4/G8, and these `(B,N)` cells
select the specialization. Other shapes remain on an exact fallback.

## Scope and next work

This is a real exact-kernel result, but it is not yet a universal A100 backend.
The matrix is one process per cell rather than a repeated paired-trial gate, and
the production-plan canary covers B4/32K. Native paged KV, ragged lengths,
FP16/FP8, and arbitrary heads remain unpromoted.

Since this D64 phase was recorded, one contiguous D128 cell has passed a strict
paired production-plan gate on both A100 SXM4 40GB and 80GB. Its distinct
register-resident K/V lifetime and narrow B4/G8/16K boundary are documented in
[Exact-Native A100 D128 Register-Resident Pipeline](exact_native_a100_d128_register_pipeline_20260901.md).

## Artifacts

```text
artifacts/gate0/tk_tensor_core_exact_grouped_phase_matrix_corrected_lightning_a100_20260831.json
artifacts/gate0/tk_tensor_core_exact_grouped_runtime_io_matrix_lightning_a100_20260831.json
artifacts/gate0/tk_tensor_core_exact_grouped_fused_direct_canary_lightning_a100_20260831.json
artifacts/gate0/tk_tensor_core_exact_grouped_fused_direct_matrix_lightning_a100_20260831.json
artifacts/gate0/tk_tensor_core_exact_grouped_preallocated_plan_canary_lightning_a100_20260831.json
```
