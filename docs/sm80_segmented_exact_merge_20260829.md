# SM80 Segmented Exact Merge

StreamAttn's A100 D128 paged-GQA kernel computes exact online-softmax partial
states in multiple producer CTAs and then merges those states. The original
merge launched one CTA per output row:

```text
merge_ctas = B * Hq
```

At `B=1`, `Hq=16`, that exposes only 16 merge CTAs. The segmented merge divides
the 128 output dimensions between `M` independent CTAs while preserving the
same FP32 partial states and exact online-softmax merge:

```text
merge_ctas = B * Hq * M
M in {1, 2, 4, 8}

m = max_i m_i
l = sum_i l_i * exp(m_i - m)
o[d] = sum_i o_i[d] * exp(m_i - m) / l
```

Each segment redundantly reads the small split-LSE vector and recomputes its
normalizer, but writes a disjoint output-dimension interval. Producer QK/PV
work and the mathematical attention result do not change. The automatic rule
chooses the smallest supported segment count that approaches 128 merge CTAs:

```text
M* = min { M in {1,2,4,8} : B * Hq * M >= 128 }
```

For the measured G8 shape this gives `M=8/4/2/1` at batch `1/2/4/8`.

## Evidence

All measurements below use BF16, D128, G8, page size 16, direct paged K/V, and
FlashInfer FA2 0.6.17. Correctness is a cross-backend exact check.

Exploratory A100 SXM4 phase maps covered HND and NHD at batch `1/2/4/8` and KV
capacity `16K/32K/64K`. Each layout produced `12/12` correct cells and `6/12`
timing wins. The wins were concentrated at low work:

```text
B1: 16K, 32K, 64K
B2: 16K, 32K
B4: 16K
```

These maps are diagnostic, not promoted evidence. FlashInfer was substantially
slower in those processes than in the strict warm-state calibration. The
strict A100 SXM4 B1/32K/HND rerun measured:

| Backend | p50 |
|---|---:|
| StreamAttn, 128 producer splits, 8 merge segments | `0.060416 ms` |
| FlashInfer FA2 | `0.057344 ms` |
| FlashInfer / StreamAttn | `0.949x` |

The output check passed with maximum absolute error `1.22e-4`, but the native
cell remains on exact external fallback. This result supersedes any inference
from the slower-baseline discovery processes.

## Falsified Alternatives

The campaign retained negative results because they locate the next bottleneck:

- At B2/64K, 18 split/merge schedules were all correct and all lost. The best
  reached only `0.843x`; merge scheduling cannot repair that cell.
- A 128-token, eight-warp producer was correct in 20/20 cells but lost every
  comparison. Its roughly 100 KiB shared-memory footprint limits residency to
  one CTA per SM and gives back the larger tile's reuse benefit.
- Reusing page descriptors from thread-local arrays made the measured native
  path slower, consistent with register-pressure costs. The code was removed.

The resulting boundary is structural:

```text
low B*N: merge occupancy can matter
large B*N: QK/PV producer throughput dominates
```

The next A100 D128 work should therefore improve the 64-token producer's
memory/MMA overlap or reduce partial-state traffic. More split-count or merge
segmentation sweeps have been falsified for the large-work cells.

## Artifacts

- `artifacts/universal_exact/sm80_d128_segmented_merge_hnd_phase.json`
- `artifacts/universal_exact/sm80_d128_segmented_merge_nhd_phase.json`
- `artifacts/universal_exact/sm80_d128_b2_64k_schedule_falsification.json`
- `artifacts/universal_exact/sm80_d128_tile128_b2_64k_ablation.json`
- `artifacts/universal_exact/sm80_d128_page_descriptor_reuse.json`
- `artifacts/universal_exact/sm80_calibration_segmented_merge_sxm4_attempt.json`
- `artifacts/universal_exact/sm80_calibration_segmented_merge_clean_sxm4.json`
