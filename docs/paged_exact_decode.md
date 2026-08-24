# Paged Exact Decode

## Status

StreamAttn has promoted Hopper exact decode paths that consume HND physical
pages and D128/G8 NHD page-16 storage directly, plus generic and
architecture-guarded Triton exact backends. No path gathers or repacks KV
during decode.

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

The Hopper specializations are intentionally guarded by measured shape:

~~~text
GPU:       H100 / SM90
dtype:     BF16
layout:    HND; NHD for page-16 D128/G8
capacity:  16K/32K/64K buckets
lengths:   page-16 positive ragged rows; page-64 full rows

page-16:   D64/G8, D128/G8, D128/G4
page-64:   D64/G8
~~~

A producer CTA owns one `(batch, kv_head, split)` group and computes a GQA
group with transposed `m64n8k16` WGMMA. G8 fills all eight WGMMA columns; G4
uses four active columns and masks the four padded columns. Page-64 maps one
physical page directly to one compute tile. Page-16 aliases the same swizzled
shared tile as `(16, D, 4)` and issues four direct physical-page copies into
it. There is no global gather or repack buffer. Every path preserves true-GQA
K/V sharing, exact online softmax, and exact split-state merging.

For NHD `[page, token, kv_head, D]`, each D128 token vector remains contiguous.
The direct producer therefore keeps 16-byte vector copies and changes only the
source row stride from `D` to `Hkv * D`. It writes the same swizzled shared K/V
tile as HND, so the WGMMA consumer, online softmax, and split merge are shared.

D128 does not keep K and V resident simultaneously. After WGMMA consumes a K
stage, the producer reuses that stage for V and the consumer performs PV before
the stage cycles. This two-phase lifetime keeps D128 inside Hopper's
shared-memory budget without reducing the 64-token compute tile.

The ragged page-16 producer derives its tile count from each request's device
length. On the final tile it predicates scores beyond that length to negative
infinity before online softmax. Missing fragments reuse the last valid physical
page only as a safe load address; every reused value is masked and contributes
zero probability. Empty split states merge with `lse=-inf`. The path therefore
preserves exact attention semantics without gathering, repacking, or launching
one kernel per sequence length.

The page-16 load economics are favorable for BF16:

~~~text
                     D64       D128
one K/V fragment    2 KiB      4 KiB
four K+V fragments  16 KiB     32 KiB
four page IDs       16 bytes   16 bytes
table / K+V bytes   0.098%     0.049%
~~~

The bytes are therefore effectively unchanged from a contiguous 64-token
tile; the risk is copy-command and synchronization overhead. The implementation
follows the same aliasing principle as NVIDIA CUTLASS's paged GQA example: one
MMA tile has a page-fragmented producer view and an unchanged consumer view.
Split selection balances producer waves against partial-state merge traffic:

~~~text
producer CTAs = batch * KV heads * splits
workspace     = O(batch * KV heads * splits * (D + 2))

D128/G8 full:    B1=128, B2=64, B4=32, B8=16
D128/G8 ragged:  same, except B8/N64K=24
D128/G4 full:    B1=32, B2=16, B4=8, B8=8
D128/G4 ragged:  B4/N32K=12, B4/N64K=16,
                  B8/N32K=12, B8/N64K=16

NHD D128/G8 full:
  B1=(64,128,64), B2=(64,64,128),
  B4=(32,32,64), B8=(32,32,32) for N=(16K,32K,64K)
NHD D128/G8 ragged:
  B1=(64,64,128), B2=(64,64,128),
  B4=(64,32,64), B8=(16,32,32)
~~~

The ragged exceptions are deliberate. For example, D128/G8 B8/N64K moves
from 256 to 384 producer CTAs: almost three H100 waves (396 CTA slots), which
reduces the long-tail effect from unequal request lengths. `C=24` costs 25%
less merge state than `C=32` while retaining the recovered latency margin.

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
`fa2`, which also won explicit backend selection for D128. Every promoted
schedule passed alternating-order paired trials; high-risk D128 boundaries
were rerun with nine trials and an independently chosen seed.

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

### Page-16 D64/G8

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

### Page-16 D64/G8 ragged rows

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

### Page-16 D128/G8

The promoted D128/G8 matrix covers B1/B2/B4/B8, 16K/32K/64K capacity,
full rows, and positive ragged rows down to one token. The automatic-route gate
sampled full, severe, and tiny profiles across all 12 batch/capacity cells:

| Evidence | Result |
|---|---:|
| Automatic-route cells | `36/36` correct and faster |
| Alternating-order trials | `180/180` StreamAttn wins |
| Median of cell paired medians | `1.75x` |
| Worst individual paired ratio | `1.075x` |
| Max StreamAttn error vs short-row FP32 reference | `1.68e-3` |
| Max FlashInfer error vs the same reference | `2.23e-3` |

An independent nine-trial B8/N64K ragged gate selected `C=24` after `C=16`
fell to statistical parity under unequal request lengths. Mild, severe, and
short profiles then won all `27/27` trials; the worst trial was `1.068x`.

Raw artifacts:

- `artifacts/gate0/paged_exact_d128_g8_page16_auto_promotion_h100.json`
- `artifacts/gate0/paged_exact_d128_g8_b8_n64_ragged_promotion_h100.json`

### Page-16 D128/G8 direct NHD

The NHD route removes the previous HND-only promotion boundary without a
transpose or repack. The automatic-route confirmation covered full and severe
ragged profiles for every B1/B2/B4/B8 and 16K/32K/64K capacity cell:

| Evidence | Result |
|---|---:|
| Automatic-route cells | `24/24` correct and faster |
| Alternating-order trials | `216/216` StreamAttn wins |
| Median of cell paired medians | `1.283x` |
| Best cell paired median | `1.967x` |
| Worst individual paired ratio | `1.058x` |
| Max absolute BF16 cross-backend error | `4.88e-4` |

The weakest cells are B8/64K, where both backends are already well occupied.
They still retained `1.058x` minimum paired margin. The raw artifacts are:

- `artifacts/gate0/paged_exact_nhd_d128_g8_phase_h100.json`
- `artifacts/gate0/paged_exact_nhd_d128_g8_auto_promotion_h100.json`

### Page-16 D128/G4

The final D128/G4 route covers seven length profiles at every B1/B2/B4/B8 and
16K/32K/64K capacity cell. The B8/N32K rows in the main matrix were superseded
by a separate automatic-route confirmation after their ragged split changed
from 8 to 12.

| Evidence | Result |
|---|---:|
| Final selected cells | `84/84` correct and faster |
| Alternating-order trials | `756/756` StreamAttn wins |
| Median of cell paired medians | `1.34x` |
| Worst individual paired ratio | `1.017x` |
| Max StreamAttn error vs short-row FP32 reference | `2.24e-3` |
| Max FlashInfer error vs the same reference | `2.25e-3` |

The weakest final cells are high-batch 64K rows, where exact decode is already
well occupied and StreamAttn's advantage is a few percent. Those cells remain
shape-guarded; this evidence is not extrapolated to other dimensions or GPUs.

Raw artifacts:

- `artifacts/gate0/paged_exact_d128_g4_page16_final_promotion_h100.json`
- `artifacts/gate0/paged_exact_d128_g4_b8_n32_promotion_h100.json`

## A100 and B200 Evidence

Portability uses separate architecture guards, not one universal kernel. The
experimental grouped Triton backend assigns one producer to
`(batch, kv_head, split)`, reads each K/V row once for all eight query heads,
uses tensor-core QK/PV tiles, maintains online softmax, and reuses the exact
split-state merge.

On A100 80GB PCIe, the head-private generic floor was `0.404 ms` versus
FlashInfer `0.116 ms` at B4/32K. True-GQA grouping reduced the StreamAttn side
substantially, and all `96/96` phase cells were correct, but no cell was
promoted. The best paired cell reached `0.992x`; the best unpaired median was
`0.909x`. This is a near-parity research boundary, not a speedup claim.

A separate architecture-native SM80 backend now handles BF16 direct-NHD
page-16 D128/G8 full rows. It stages four physical pages at a time with
`cp.async`, shares each K/V tile across all eight query heads in the GQA group,
uses `m16n8k16` BF16 MMA for QK and PV, maintains online-softmax state, and
merges FP32 split states exactly. It consumes paged cache tensors directly;
there is no page gather or NHD-to-HND repack in the timed path.

At B1/32K on A100 80GB SXM4, five repeated cells measured `1.185x-1.206x`
with `45/45` paired wins, and a separate 15-trial confirmation measured
`1.262x`. An independent warm-state sweep also produced a `0.975x` result
because the selected FlashInfer FA2 baseline moved from roughly `0.084 ms` to
`0.061 ms`, while StreamAttn stayed near `0.065 ms`. B2/B4/B8 lose. Therefore
this backend remains explicit experimental opt-in and is not part of automatic
dispatch. Its next scaling problem is the PV shared-to-register transpose, not
more split-count tuning.

On B200, all `108/108` grouped phase cells were correct, but none beat the
fastest correct FlashInfer backend. The best paired cell reached `0.667x`.
Blackwell makes the non-MMA pipeline decisive: the next backend must use paged
TMA issue, TMEM, fully asynchronous MMA, and overlapped softmax/epilogue rather
than more split-count tuning. NVIDIA CUTLASS example 93 is the implementation
reference for that path.

Raw artifacts:

- `artifacts/gate0/paged_exact_nhd_d128_g8_grouped_phase_a100.json`
- `artifacts/gate0/paged_exact_sm80_cp_async_phase_a100.json`
- `artifacts/gate0/paged_exact_sm80_cp_async_split_sweep_a100.json`
- `artifacts/gate0/paged_exact_sm80_cp_async_winner_sweep_a100.json`
- `artifacts/gate0/paged_exact_sm80_cp_async_repro_b1_a100.json`
- `artifacts/gate0/paged_exact_sm80_cp_async_confirm_b1_a100.json`
- `artifacts/gate0/paged_exact_nhd_d128_g8_grouped_phase_b200.json`

## Benchmark

Run a paired benchmark against FlashInfer:

~~~bash
python benchmarks/profile_paged_exact_decode.py \
  --batch 4 \
  --kv-len 32768 \
  --q-heads 16 \
  --kv-heads 2 \
  --head-dim 128 \
  --page-size 16 \
  --layout NHD \
  --dtype bf16 \
  --flashinfer-backends auto,fa2,fa3,trtllm-gen \
  --output-json artifacts/paged_exact_b4_32k_d128_g8_h100.json
~~~

The A100 architecture-native candidate is deliberately explicit:

~~~bash
python benchmarks/profile_paged_exact_decode.py \
  --batch 1 \
  --kv-len 32768 \
  --q-heads 16 \
  --kv-heads 2 \
  --head-dim 128 \
  --page-size 16 \
  --layout NHD \
  --dtype bf16 \
  --splits 128 \
  --sm80-cp-async-experimental \
  --flashinfer-backends auto,fa2
~~~

The benchmark randomizes physical page order, checks output parity, reuses both
plans, and reports paired raw samples. When several FlashInfer backends are
requested it selects the fastest correct initial median, rather than assuming
that `auto` is the strongest baseline. Packing time is excluded for both
backends because page construction belongs to cache management, not decode.

For rows no longer than 2,048 tokens, the harness also materializes an
independent FP32 dense reference. This matters in BF16: two valid reduction
orders can differ by one representable output step (`0.00390625`) even when
each is within roughly `0.0023` of FP32.

## Promotion Gate

A cell can be promoted only when:

~~~text
correctness passes against a cross-backend or independent FP32 reference
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
D128/G4 and D128/G8, HND, page-16
D128/G8, NHD, page-16
~~~

Other dimensions, page sizes, variable page-64 lengths, and non-Hopper GPUs
remain on generic or explicitly experimental exact backends until separately
measured and promoted. A100 has one variable B1 candidate but no promoted
phase; B200 has a measured negative phase diagram. Successful compilation or
correctness alone does not enable either architecture's experimental routes.
