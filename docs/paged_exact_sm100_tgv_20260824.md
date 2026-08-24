# Native SM100 Paged Exact Decode

StreamAttn's first Blackwell-native exact backend targets one deliberately
narrow serving shape:

```text
GPU: B200 / SM100a
dtype: BF16
Q: [B, 1, 16, 128]
K/V pages: separate NHD [pages, 16, 2, 128]
GQA group: 8
rows: full 32K or 64K
```

The implementation is adapted from NVIDIA CUTLASS example 93 at commit
`7107b05535f8977f5ecb9d01ee203205b1fd9bc4`. The vendored example headers retain
their BSD-3-Clause notices. StreamAttn changes the host path to consume separate
NHD K and V page tensors directly, binds it to PyTorch's current CUDA stream,
and exposes calibrated cluster split counts through `PagedExactDecodePlan`.

## Why The Generic Path Lost

The earlier grouped Triton path shared K/V across all eight query heads and was
correct in 108/108 B200 cells, but its best paired result was only `0.667x`
against FlashInfer. The algebra was right; the pipeline was not Blackwell
native. On SM100, faster MMA increases the relative cost of page issue,
shared-memory movement, synchronization, softmax rescaling, and the epilogue.

The native path changes the dataflow:

```text
paged NHD K/V -> TMA -> shared tiles -> tcgen05 MMA -> TMEM
             -> online softmax -> cluster split reduction -> output
```

It does not gather pages into contiguous KV, combine K and V, or repack NHD to
HND in the timed path. The page table is padded once when the fixed-buffer plan
is created because the asynchronous metadata stage can read ahead by 64 slots.

## Confirmed Cells

The independent boundary run used randomized physical pages, FlashInfer 0.6.17,
30 initial samples, and 15 alternating-order paired trials per cell.

| B | N | Cluster splits | StreamAttn ms | FlashInfer ms | Paired median | Paired min | Wins |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32K | 16 | 0.02110 | 0.02877 | 1.320x | 1.276x | 15/15 |
| 2 | 32K | 16 | 0.02128 | 0.03066 | 1.443x | 1.388x | 15/15 |
| 2 | 64K | 16 | 0.03325 | 0.04539 | 1.369x | 1.337x | 15/15 |
| 4 | 32K | 8 | 0.03366 | 0.04509 | 1.377x | 1.349x | 15/15 |
| 4 | 64K | 8 | 0.05597 | 0.06464 | 1.165x | 1.125x | 15/15 |
| 8 | 32K | 4 | 0.05570 | 0.06373 | 1.155x | 1.122x | 15/15 |

All eight confirmation cells were correct; maximum BF16 cross-backend error
was `2.44e-4`. Only the six rows above are promoted. The measured boundaries
remain explicit:

```text
B1 / 64K: paired median 0.982x, 3/15 wins -> fallback
B8 / 64K: paired median 1.006x, min 0.989x -> fallback
```

## Dispatch Contract

Automatic routing additionally requires compute capability 10.0, BF16, direct
contiguous NHD pages, page size 16, D128, Hq16/Hkv2, integer page metadata, and
full fixed-length rows. A current CUTLASS checkout must be available through
`STREAMATTN_SM100_CUTLASS_ROOT`, `STREAMATTN_CUTLASS_ROOT`, `CUTLASS_ROOT`,
`CUTLASS_PATH`, or `/opt/cutlass`. If any invariant or header dependency is
missing, planning fails closed to the generic exact backend.

Evidence:

- `artifacts/gate0/paged_exact_sm100_tgv_arch_phase_b200.json`
- `artifacts/gate0/paged_exact_sm100_tgv_confirmation_b200.json`
- `artifacts/gate0/paged_exact_sm100_tgv_auto_route_b200.json`
- `artifacts/gate0/sm100_cutlass_paged_gqa_floor_b200.json`

Primary implementation reference:

- <https://github.com/NVIDIA/cutlass/tree/main/examples/93_blackwell_low_latency_gqa>
