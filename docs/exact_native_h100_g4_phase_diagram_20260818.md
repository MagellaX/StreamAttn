# H100 exact-native G4 phase diagram (2026-08-18)

## Result

StreamAttn's native BF16 exact-decode kernel now supports a second true-GQA
shape family on H100:

```text
Hq=16, Hkv=4, G=4, D=64
N=16K/32K/64K
B=1/2/4/8/16, except B1/64K
head-major contiguous KV
```

This is exact attention over every KV token. It is not the seed-only route and
does not use FlashInfer or FlashAttention internally. FlashInfer 0.6.12 is the
paired benchmark reference.

The public `StreamAttnExactNativeDirectRunner` gate produced:

| Batch | KV length | Splits | StreamAttn vs FlashInfer | Paired minimum | Wins |
|---:|---:|---:|---:|---:|---:|
| 1 | 16K | 64 | 1.449x | 1.438x | 9/9 |
| 1 | 32K | 128 | 1.120x | 1.115x | 9/9 |
| 2 | 16K | 64 | 1.143x | 1.113x | 9/9 |
| 2 | 32K | 64 | 1.027x | 1.022x | 9/9 |
| 2 | 64K | 64 | 1.038x | 1.034x | 9/9 |
| 4 | 16K | 32 | 1.048x | 1.041x | 9/9 |
| 4 | 32K | 32 | 1.053x | 1.052x | 9/9 |
| 4 | 64K | 32 | 1.060x | 1.059x | 9/9 |
| 8 | 16K | 16 | 1.049x | 1.046x | 9/9 |
| 8 | 32K | 16 | 1.062x | 1.059x | 9/9 |
| 8 | 64K | 16 | 1.064x | 1.064x | 9/9 |
| 16 | 16K | 8 | 1.062x | 1.061x | 9/9 |
| 16 | 32K | 8 | 1.065x | 1.063x | 9/9 |
| 16 | 64K | 8 | 1.075x | 1.074x | 9/9 |

Every promoted cell passed exact-reference checks and two independent paired
H100 processes before the public serving gate. The primary B2/B4/B8 matrices
alone recorded 162/162 paired wins across the two independent runs.

## Kernel finding

The producer retains the Hopper `m64n8k16` WGMMA atom. G4 supplies four real
query-head columns and zero-fills four inactive columns inside the CUDA kernel:

```text
WGMMA column utilization = active heads / physical columns = 4 / 8 = 50%
```

Despite the wasted tensor-core columns, G4 still wins. Doubling the number of KV
groups relative to G8 doubles independent producer groups and reaches the useful
global work floor with half as many splits:

```text
producer CTAs = B * Hkv * Csplit
```

Most promoted cells select approximately 512 producer CTAs:

```text
B2  -> C=64
B4  -> C=32
B8  -> C=16
B16 -> C=8
```

This is evidence that, for this decode regime, occupancy and context-axis
mapping can outweigh 50% WGMMA column utilization. A dedicated G4 MMA path is
therefore an optimization opportunity, not a prerequisite for exact victory.

## Boundary evidence

B1/64K is deliberately not promoted. A dense split sweep over
`C={32,48,64,80,96,112,128,160}` produced a best paired result of `0.985x`
with 0/9 wins in the confirmation process. The public router therefore leaves
that cell on another exact backend.

The split optimum is discrete rather than monotonic. At B1/16K, `C=64` wins
because the merge cost of larger split counts exceeds the remaining producer
gain. At B1/32K, `C=128` wins. The serving registry records measured cells and
does not interpolate.

## Evidence artifacts

- `artifacts/gate0/sm90_exact_g4_phase_diagram_h100_20260818.json`
- `artifacts/gate0/sm90_exact_g4_extended_confirmation_h100_20260818.json`
- `artifacts/gate0/sm90_exact_g4_b1_frontier_h100_20260818.json`
- `artifacts/gate0/sm90_exact_g4_b16_confirmation_h100_20260818.json`
- `artifacts/gate0/sm90_exact_g4_serving_gate_h100_20260818.json`
- `artifacts/gate0/transposed_wgmma_exact_g8_regression_h100_20260818.json`

The final G8 B4/32K regression gate passed exact output, 9/9 paired trials,
`1.026x` median kernel speedup, and `1.026x` public serving speedup.

## Scope limits

This evidence is specific to H100/SM90, BF16, D64, Hq16/Hkv4, contiguous
head-major KV, and the listed batch/context cells. It is not evidence for D128,
FP16, paged native KV, A100, B200, or arbitrary GQA shapes. Those remain separate
kernel and calibration work.
