# Paged Exact Decode

## Status

StreamAttn has an experimental exact decode path that consumes a paged KV cache
directly. It is wired into the public engine but is not a promoted performance
route until paired H100 measurements pass.

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

## Kernel

The Triton backend launches split-K producers over:

~~~text
[batch, query_heads, splits]
~~~

Each producer:

1. Loads the query head once.
2. Walks its logical page interval.
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

The first matrix is H100, BF16:

~~~text
B = 1, 2, 4, 8
N = 16K, 32K, 64K
D64/G4
D64/G8
~~~

D128/G4 follows after the D64 phase diagram identifies whether page-table
lookup, duplicated GQA reads, or split-state merge is the limiting term.
