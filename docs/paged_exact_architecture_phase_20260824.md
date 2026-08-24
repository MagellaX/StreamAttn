# Paged Exact Decode Across SM80, SM90, and SM100

This phase answers two portability questions for exact paged decode:

1. Can StreamAttn consume page-16 NHD cache storage directly on H100 without
   repacking it to HND?
2. Does the same true-GQA grouped computation become competitive on A100 and
   B200 without architecture-specific pipelines?

The measured shape is BF16, D128, 16 query heads, 2 KV heads, GQA group 8,
page size 16, with full and severe ragged rows where noted. FlashInfer's
fastest correct requested backend is selected per run.

## Result

| GPU | StreamAttn path | Correct cells | Winning cells | Strongest measured result | Decision |
|---|---|---:|---:|---:|---|
| H100 80GB | Direct NHD SM90 WGMMA | 24/24 auto cells | 24/24 | 1.283x median cell; 1.058x worst paired trial | Promoted for measured cells |
| A100 80GB PCIe | SM80 grouped Triton floor | 96/96 | 0/96 | 0.992x best paired cell | Experimental; near parity |
| B200 | SM100 grouped Triton floor | 108/108 | 0/108 | 0.667x best paired cell | Experimental; pipeline redesign required |

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

On B200, grouped algebra alone is much farther from the baseline. Faster MMA
makes page issue, synchronization, exponentiation, rescaling, and epilogue
costs proportionally larger. The next SM100 backend must be a Blackwell-native
pipeline using paged TMA issue, tensor memory, fully asynchronous MMA, and
overlapped softmax/epilogue. Split-count tuning cannot repair that mismatch.

## Dispatch Policy

```text
SM90 + measured NHD page-16 D128/G8 cell:
    direct NHD WGMMA backend

SM80 grouped backend:
    explicit experimental opt-in only

SM100 grouped backend:
    explicit experimental opt-in only

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
- `artifacts/gate0/paged_exact_nhd_d128_g8_grouped_phase_b200.json`
