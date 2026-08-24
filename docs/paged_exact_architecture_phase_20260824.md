# Paged Exact Decode Across SM80, SM90, and SM100

This phase answers two portability questions for exact paged decode:

1. Can StreamAttn consume page-16 NHD cache storage directly on H100 without
   repacking it to HND?
2. Does the same true-GQA grouped computation become competitive on A100 and
   B200 without architecture-specific pipelines?
3. When it does not, can architecture-native Ampere and Blackwell pipelines
   recover an exact-kernel advantage without changing the paged cache layout?

The measured shape is BF16, D128, 16 query heads, 2 KV heads, GQA group 8,
page size 16, with full and severe ragged rows where noted. FlashInfer's
fastest correct requested backend is selected per run.

## Result

| GPU | StreamAttn path | Correct cells | Winning cells | Strongest measured result | Decision |
|---|---|---:|---:|---:|---|
| H100 80GB | Direct NHD SM90 WGMMA | 24/24 auto cells | 24/24 | 1.283x median cell; 1.058x worst paired trial | Promoted for measured cells |
| A100 80GB PCIe | SM80 grouped Triton floor | 96/96 | 0/96 | 0.992x best paired cell | Experimental; near parity |
| A100 80GB SXM4 | Native SM80 `cp.async` + MMA | B1/B2/B4/B8 at 32K | Variable | B1 repeated `1.20x-1.26x`; one sweep `0.975x` | Correct candidate; not promoted |
| B200 | Native SM100 TMA+TMEM+`tcgen05` | 8/8 independent confirmation cells | 6/8 | 1.122x worst paired trial among promoted cells; 1.443x best paired median | Six full-row NHD cells promoted |

The H100 gate includes B1/B2/B4/B8, 16K/32K/64K capacity, and full plus
severe ragged profiles. It won all 216 alternating-order paired trials with a
maximum BF16 cross-backend error of `4.88e-4`.

## Direct NHD Result

NHD stores pages as:

```text
[physical_page, token_in_page, kv_head, head_dim]
```

Each D128 token vector remains contiguous, but adjacent tokens for one KV head
have stride `Hkv * D`. The SM90 producer therefore computes the physical page
address directly and copies strided token rows into the existing WGMMA shared
tile:

```text
page_base = ((physical_page * 16 * Hkv) + kv_head) * D
token_stride = Hkv * D
```

There is no NHD-to-HND transpose, global gather, or temporary page buffer in
the timed path. Once staged in shared memory, the existing transposed WGMMA
consumer and online-softmax merge are unchanged.

## Architecture Math

The generic head-private path rereads shared GQA K/V for every query head. A
grouped path assigns one producer to `(batch, kv_head, split)` and evaluates
all query heads sharing that KV head together. Its ideal KV traffic reduction
is:

```text
bytes_grouped / bytes_head_private ~= 1 / G
```

For GQA group 8, that is an 8x reduction. Tensor-core row packing introduces
the countervailing cost:

```text
row_utilization = active_query_rows / tensor_core_tile_rows
                = 8 / 16
                = 50%
```

The useful route criterion is therefore not only the byte ratio. It is:

```text
T_grouped = T_page_issue + T_QK/PV(G padded to 16)
          + T_online_softmax + T_split_merge

promote only if T_grouped < T_fastest_correct_baseline
```

On A100, removing repeated KV reads recovered most of the generic path's
deficit, but the best paired result remained just below parity. The next SM80
backend needs explicit `cp.async` staging, MMA-friendly shared layouts, and a
register/occupancy design measured independently from SM90.

That backend now exists for BF16 direct-NHD page-16 D128/G8. One CTA owns
`(batch, kv_head, split)`, stages 64-token K/V tiles with `cp.async`, evaluates
all eight query heads with `m16n8k16` BF16 MMA, updates online-softmax state,
and merges FP32 split states exactly. It does not gather or repack pages.

The short-reference gate passed with `9.77e-4` maximum error; every 32K phase
cell stayed at or below `2.44e-4` cross-backend error. The first B1 phase run
measured `1.29x`, five repeated B1 cells measured `1.185x-1.206x` with `45/45`
paired wins, and a final 15-trial confirmation measured `1.262x`. However, a
separate warm-state sweep measured a faster FlashInfer FA2 baseline and only
`0.975x` at the same B1/C128 cell. StreamAttn itself stayed near
`0.065-0.067 ms`; FlashInfer varied from roughly `0.061-0.084 ms`.

This is evidence for the architecture and an exact-kernel candidate, not an
automatic A100 promotion. B2/B4/B8 are also excluded. The current scaling
limit is the PV shared-to-register edge: QK uses native `ldmatrix`, while the
correct PV path uses scalar loads from a logical transposed V view. CuTe
rejected both normal and transposed `ldmatrix` atoms for the tested physical V
layouts. A production B4+ backend needs a hand-authored transposed load map or
a producer layout designed around `ldmatrix.trans`.

On B200, grouped algebra alone was much farther from the baseline. Faster MMA
makes page issue, synchronization, exponentiation, rescaling, and epilogue
costs proportionally larger. That negative result motivated a separate native
SM100 pipeline adapted from NVIDIA CUTLASS example 93. The backend issues paged
NHD K/V through TMA, stores accumulators in TMEM, uses asynchronous `tcgen05`
MMA and cluster reduction, and preserves online-softmax split merging.

The architecture phase passed all 12 B1/B2/B4/B8 by 16K/32K/64K cells. A
second run independently confirmed the 32K/64K boundary with 15 paired trials
per cell. Six cells won all `90/90` paired trials collectively and now route
automatically with calibrated split counts: C16 for B1/32K and B2/32K/64K, C8
for B4/32K/64K, and C4 for B8/32K. B1/64K and B8/64K remain fallbacks. The
backend reads separate page-16 NHD K/V directly; there is no combined-KV copy
or layout repack in the timed path.

## Dispatch Policy

```text
SM90 + measured NHD page-16 D128/G8 cell:
    direct NHD WGMMA backend

SM80 grouped backend:
    explicit experimental opt-in only

SM80 cp.async + MMA backend:
    explicit experimental opt-in only; full NHD page-16 D128/G8 rows

SM100 native TGV backend:
    automatic only for the six confirmed full-row NHD D128/G8 cells

SM100 grouped Triton backend and all other SM100 cells:
    explicit experimental opt-in or generic exact fallback

all unsupported or unpromoted cells:
    generic exact backend
```

Correctness does not imply promotion. A backend enters automatic routing only
after paired timing, alternating launch order, cross-backend correctness, and
no-repack verification pass for that exact architecture and shape cell.

## Evidence

- `artifacts/gate0/paged_exact_nhd_d128_g8_phase_h100.json`
- `artifacts/gate0/paged_exact_nhd_d128_g8_auto_promotion_h100.json`
- `artifacts/gate0/paged_exact_nhd_d128_g8_floor_a100.json`
- `artifacts/gate0/paged_exact_nhd_d128_g8_grouped_phase_a100.json`
- `artifacts/gate0/paged_exact_sm80_cp_async_phase_a100.json`
- `artifacts/gate0/paged_exact_sm80_cp_async_split_sweep_a100.json`
- `artifacts/gate0/paged_exact_sm80_cp_async_winner_sweep_a100.json`
- `artifacts/gate0/paged_exact_sm80_cp_async_repro_b1_a100.json`
- `artifacts/gate0/paged_exact_sm80_cp_async_confirm_b1_a100.json`
- `artifacts/gate0/paged_exact_nhd_d128_g8_grouped_phase_b200.json`
- `artifacts/gate0/paged_exact_sm100_tgv_arch_phase_b200.json`
- `artifacts/gate0/paged_exact_sm100_tgv_confirmation_b200.json`
