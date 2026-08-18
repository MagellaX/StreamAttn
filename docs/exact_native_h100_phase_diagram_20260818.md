# Exact-Native H100 D64/G8 Phase Diagram

Date: 2026-08-18

## Question

The first exact-native win covered one cell: H100, BF16, D64, GQA group size
eight, `B4`, and 32K KV. This study tests whether that result is a useful
hardware region or a single-cell optimum. StreamAttn and FlashInfer both
compute exact attention over every KV token.

Each cell used native head-major KV, 20 warmups, 200 timed iterations, nine
repeated samples, and nine alternating paired comparisons. StreamAttn searched
three split counts around the producer occupancy target. A robust win requires
all nine paired trials to win and the weakest trial to remain above `1.0x`.

## Replicated 3x3 map

Two fresh H100 80GB processes independently measured
`B={2,4,8} x N={16K,32K,64K}`.

| B | KV | Run 1 median/min | Run 2 median/min | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 2 | 16K | `1.3468x / 1.3306x` | `1.4264x / 1.3995x` | strong repeated win |
| 2 | 32K | `1.0159x / 0.9914x` | `1.0596x / 1.0350x` | environment-sensitive boundary |
| 2 | 64K | `0.9744x / 0.9732x` | `0.9604x / 0.9597x` | repeated loss |
| 4 | 16K | `1.0999x / 1.0954x` | `1.3234x / 1.2179x` | strong repeated win |
| 4 | 32K | `1.0283x / 1.0254x` | `0.9951x / 0.9924x` | narrow/volatile legacy cell |
| 4 | 64K | `1.0396x / 1.0383x` | `1.0041x / 1.0028x` | repeated narrow win |
| 8 | 16K | `1.0444x / 1.0394x` | `1.0104x / 1.0091x` | repeated win |
| 8 | 32K | `1.0540x / 1.0514x` | `1.0290x / 1.0277x` | repeated win |
| 8 | 64K | `1.0813x / 1.0795x` | `1.0530x / 1.0519x` | repeated win |

Both processes produced seven robust wins out of nine, but the marginal cell
changed between `B2/32K` and `B4/32K`. The stable all-trial intersection is six
cells. This demonstrates a real D64/G8 region while also showing why static
performance claims need cell-level calibration.

All cells remained finite and deterministic. Maximum FP32-reference error
across the final serving gate was `1.56e-4`, inside the established `5e-4`
BF16 exact-kernel gate.

## B2 context frontier

A third fresh process mapped the unexpected low-batch advantage more densely:

| KV | Best splits | Producer CTAs | Median speedup | Minimum | Wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 64 | 256 | `1.5798x` | `1.5608x` | `9/9` |
| 12K | 64 | 256 | `1.3211x` | `1.3183x` | `9/9` |
| 16K | 64 | 256 | `1.2315x` | `1.2268x` | `9/9` |
| 20K | 64 | 256 | `1.2064x` | `1.2006x` | `9/9` |
| 24K | 64 | 256 | `1.0945x` | `1.0880x` | `9/9` |
| 28K | 64 | 256 | `1.0940x` | `1.0872x` | `9/9` |
| 32K | 128 | 512 | `1.0498x` | `1.0476x` | `9/9` |
| 40K | 128 | 512 | `1.0126x` | `0.9885x` | `8/9` |
| 48K | 128 | 512 | `0.9832x` | `0.9816x` | `0/9` |
| 56K | 128 | 512 | `0.9970x` | `0.9964x` | `0/9` |
| 64K | 128 | 512 | `0.9972x` | `0.9959x` | `0/9` |

The measured low-batch frontier is therefore robust through 32K, ambiguous at
40K, and negative from 48K onward in this process.

## Split-count conclusion

The governing scheduler variables are:

```text
producer_ctas   = B * Hkv * Csplit
tiles_per_split = ceil((N / 64) / Csplit)
workspace       = B * Hkv * Csplit * (8 * 64 + 8) * sizeof(float)
```

Increasing `Csplit` reduces producer work in quantized 64-token steps but grows
the exact split-state merge and workspace. A dedicated long-context sweep used
`C={64,96,128,160,192,224,256}` at `B2` and 40K-64K. It produced `0/4`
robust wins. More splitting did not recover the long-context loss; the merge
and scheduling costs outweighed producer savings.

This falsifies the simple rule "keep increasing splits until occupancy wins."
The actual optimization is:

```text
min_C T_producer(B, N, C) + T_merge(B, C)
subject to exact output and enough producer waves
```

## Guarded backend cells

The serving backend now keeps a discrete split table rather than interpolating
between shapes:

```text
B2 / 16K -> C64
B4 / 16K -> C64
B4 / 32K -> C64  (legacy narrow cell)
B4 / 64K -> C64
B8 / 16K -> C32
B8 / 32K -> C32
B8 / 64K -> C32
```

`B2/40K+` remains excluded. Other shapes retain StreamAttn's exact fallback.
The map intentionally does not promote the single-run `B2/8K-28K` cells until
they receive a second process-level replication.

## Public serving gate

The final H100 run exercised `StreamAttnExactNativeDirectRunner`, not only the
raw plan. Every promoted cell selected `sm90_transposed_gqa_wgmma_exact`, used
the calibrated split count, and passed all nine paired trials:

| B | KV | Splits | Serving median | Minimum | Wins |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 16K | 64 | `1.4317x` | `1.4158x` | `9/9` |
| 4 | 16K | 64 | `1.1826x` | `1.1767x` | `9/9` |
| 4 | 32K | 64 | `1.0245x` | `1.0228x` | `9/9` |
| 4 | 64K | 64 | `1.0358x` | `1.0345x` | `9/9` |
| 8 | 16K | 32 | `1.0528x` | `1.0516x` | `9/9` |
| 8 | 32K | 32 | `1.0506x` | `1.0495x` | `9/9` |
| 8 | 64K | 32 | `1.0828x` | `1.0822x` | `9/9` |

This is the promoted exact-native region. It is separate from StreamAttn's
seed-only modes: no KV tokens are omitted in any row above.

## Artifacts

```text
artifacts/gate0/transposed_wgmma_exact_phase_diagram_h100_20260818.json
artifacts/gate0/transposed_wgmma_exact_phase_diagram_repeat2_h100_20260818.json
artifacts/gate0/transposed_wgmma_exact_b2_context_frontier_h100_20260818.json
artifacts/gate0/transposed_wgmma_exact_b2_long_split_sweep_h100_20260818.json
artifacts/gate0/transposed_wgmma_exact_promoted_serving_gate_h100_20260818.json
```
