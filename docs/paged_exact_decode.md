# Paged Exact Decode

## Status

StreamAttn has a promoted Hopper exact decode path that consumes HND physical
pages directly, plus a generic Triton exact fallback for other supported paged
layouts. Neither path gathers or repacks KV during decode.

## Contract

The query is [batch, 1, query_heads, head_dim]. Physical pages use either:

~~~text
NHD: [num_pages, page_size, kv_heads, head_dim]
HND: [num_pages, kv_heads, page_size, head_dim]
~~~

Each request supplies:

~~~text
page_table:       [batch, max_pages_per_request]
sequence_lengths: [batch]
~~~

Trailing page-table slots may contain -1. Every active slot must reference a
valid physical page. Planning validates the initial metadata once; serving code
may update the same buffers in place while preserving those invariants.

## Kernels

### Promoted SM90 path

The Hopper specializations are intentionally narrow:

~~~text
GPU:       H100 / SM90
dtype:     BF16
Q heads:   16
KV heads:  2
head dim:  64
layout:    HND
page size: 16 or 64
capacity:  16K/32K/64K buckets
lengths:   page-16 positive ragged rows; page-64 full rows
~~~

A producer CTA owns one `(batch, kv_head, split)` group and computes all eight
query heads together with transposed `m64n8k16` WGMMA. Page-64 maps one physical
page directly to one compute tile. Page-16 aliases the same swizzled shared
tile as `(16, D, 4)` and issues four direct physical-page copies into it. There
is no global gather or repack buffer. Both paths preserve true-GQA K/V sharing,
exact online softmax, and exact split-state merging.

The ragged page-16 producer derives its tile count from each request's device
length. On the final tile it predicates scores beyond that length to negative
infinity before online softmax. Missing fragments reuse the last valid physical
page only as a safe load address; every reused value is masked and contributes
zero probability. Empty split states merge with `lse=-inf`. The path therefore
preserves exact attention semantics without gathering, repacking, or launching
one kernel per sequence length.

The page-16 load economics are favorable for D64 BF16:

~~~text
one K or V fragment = 16 * 64 * 2 bytes = 2 KiB
four K+V fragments  = 4 * 2 * 2 KiB   = 16 KiB
four page IDs       = 4 * 4 bytes       = 16 bytes
page-table overhead / K+V bytes          = 0.098%
~~~

The bytes are therefore effectively unchanged from a contiguous 64-token
tile; the risk is copy-command and synchronization overhead. The implementation
follows the same aliasing principle as NVIDIA CUTLASS's paged GQA example: one
MMA tile has a page-fragmented producer view and an unchanged consumer view.
The selected split table then keeps producer parallelism near the H100 occupancy
region:

~~~text
producer CTAs = batch * KV heads * splits
B1-B4: 64 splits
B8:    32 splits
~~~

Reference implementation pattern:
[CUTLASS paged GQA](https://github.com/NVIDIA/cutlass/blob/main/examples/93_blackwell_low_latency_gqa/tgv_gqa_paged.cuh).

### Generic Triton path

The Triton backend launches split-K producers over:

~~~text
[batch, query_heads, splits]
~~~

Each head-private producer:

1. Loads the query head once.
2. Walks its logical page interval in multi-page token tiles.
3. Loads the physical page ID from the block table.
4. Reads K/V directly from that page.
5. Maintains FP32 online-softmax state (m, l, numerator).
6. Writes one compact partial state.

A second kernel merges partial states with the exact online-softmax merge:

~~~text
m = max_i(m_i)
l = sum_i(l_i * exp(m_i - m))
n = sum_i(n_i * exp(m_i - m))
out = n / l
~~~

No per-request contiguous KV tensor is created. Workspace and output buffers
are allocated once by PagedExactDecodePlan.

## H100 Evidence

Paired on an H100 80GB with randomized physical page order. Page-64 used
FlashInfer 0.6.12. The current page-16 gate uses version-matched
`flashinfer-python`/`flashinfer-cubin` 0.6.17; `backend="auto"` resolved to
`fa2`. Each promoted cell passed nine alternating-order paired trials and the
exact output tolerance.

### Page-64

Paired median speedups:

| Batch | 16K | 32K | 64K |
|---:|---:|---:|---:|
| 1 | `2.17x` | `2.24x` | `1.46x` |
| 2 | `2.09x` | `1.79x` | `1.46x` |
| 4 | `1.82x` | `1.50x` | `1.31x` |
| 8 | `1.53x` | `1.32x` | `1.21x` |

All 12 cells won `9/9` trials. The minimum paired ratio observed across the
matrix was `1.19x`; max absolute output error was at most `2.44e-4` in BF16.
The raw artifact is
`artifacts/gate0/paged_sm90_exact_strict_matrix_h100_20260822.json`.

### Page-16

Paired median speedups:

| Batch | 16K | 32K | 64K |
|---:|---:|---:|---:|
| 1 | `2.07x` | `1.80x` | `1.43x` |
| 2 | `2.06x` | `1.74x` | `1.51x` |
| 4 | `1.93x` | `1.52x` | `1.35x` |
| 8 | `1.57x` | `1.35x` | `1.21x` |

All 12 full-row controls won `9/9` trials (`108/108` total) against FlashInfer
0.6.17. The minimum paired ratio was `1.20x`; max absolute output error was at
most `2.44e-4` in BF16.

### Page-16 ragged rows

The ragged gate covers tail-only, mixed mild/severe, short-heavy, uniform
`N/8`, uniform `N/64`, and one-token request profiles. Across the promoted
B1/B2/B4/B8 and 16K/32K/64K capacity matrix:

| Evidence | Result |
|---|---:|
| Ragged cells | `76/76` correct and faster |
| Alternating-order trials | `684/684` StreamAttn wins |
| Median of cell paired medians | `2.04x` |
| Worst individual paired ratio | `1.17x` |
| Max absolute BF16 error | `1.95e-3` |
| One-token endpoint | `0` max error; all B1-B8 cells faster |

The larger worst-case error occurs in the very short `N/64` profile; every
cell remains below the benchmark's `1e-2` BF16 exact-output tolerance. Long
tail/mild/severe profiles stay at or below `4.88e-4`.

Raw page-16 artifacts:

- `artifacts/gate0/paged_exact_page16_selected_b1_b4_h100.json`
- `artifacts/gate0/paged_exact_page16_selected_b8_h100.json`
- `artifacts/gate0/paged_exact_page16_ragged_phase_h100_flashinfer_0_6_17.json`
- `artifacts/gate0/paged_exact_page16_ragged_short_boundary_h100_flashinfer_0_6_17.json`
- `artifacts/gate0/paged_exact_page16_ragged_utilization_boundary_h100_flashinfer_0_6_17.json`
- `artifacts/gate0/paged_exact_page16_ragged_minimum_h100_flashinfer_0_6_17.json`

## Benchmark

Run a paired benchmark against FlashInfer:

~~~bash
python benchmarks/profile_paged_exact_decode.py \
  --batch 4 \
  --kv-len 32768 \
  --q-heads 16 \
  --kv-heads 2 \
  --head-dim 64 \
  --page-size 16 \
  --layout HND \
  --dtype bf16 \
  --output-json artifacts/paged_exact_b4_32k_d64_g8_h100.json
~~~

The benchmark randomizes physical page order, checks output parity, reuses both
plans, and reports paired raw samples. Packing time is excluded for both
backends because page construction belongs to cache management, not decode.

## Promotion Gate

A cell can be promoted only when:

~~~text
correctness passes against an exact reference
paired median speedup > 1.0
at least 7 of 9 paired trials win
no page-to-contiguous copy occurs in the timed path
workspace and output buffers are reused
~~~

The promoted matrix is H100, BF16:

~~~text
B = 1, 2, 4, 8
N = 16K, 32K, 64K
D64/G8, HND, page-16 and page-64
~~~

Other page sizes, variable page-64 lengths, and NHD WGMMA remain on the generic
exact backend until separately measured and promoted.
