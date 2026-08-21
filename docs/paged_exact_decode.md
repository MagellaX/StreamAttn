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

The Hopper specialization is intentionally narrow:

~~~text
GPU:       H100 / SM90
dtype:     BF16
Q heads:   16
KV heads:  2
head dim:  64
layout:    HND
page size: 64
lengths:   full fixed 16K/32K/64K buckets
~~~

One physical page is one 64-token WGMMA tile. A producer CTA owns one
`(batch, kv_head, split)` group, looks up each physical page once, and computes
all eight query heads together with transposed `m64n8k16` WGMMA. This preserves
true-GQA K/V sharing across the query group while retaining exact online
softmax and split-state merging.

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

Paired against FlashInfer 0.6.12 on an H100 80GB with randomized physical page
order. Each promoted cell passed nine alternating-order paired trials and the
exact output tolerance. The values below are paired median speedups:

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

## Benchmark

Run a paired benchmark against FlashInfer:

~~~bash
python benchmarks/profile_paged_exact_decode.py \
  --batch 4 \
  --kv-len 32768 \
  --q-heads 16 \
  --kv-heads 2 \
  --head-dim 64 \
  --page-size 64 \
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
D64/G8, HND, page-64
~~~

Page-16 WGMMA is the next paged backend target. It must stage four physical
pages into each 64-token WGMMA tile without introducing a gather buffer or
losing asynchronous copy overlap. Until that gate passes, page-16 uses the
generic exact path and carries no performance claim.
