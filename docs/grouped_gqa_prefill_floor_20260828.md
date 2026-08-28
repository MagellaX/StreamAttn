# KV-Group-Owned Exact GQA Prefill Floor

## Question

The first compact GQA prefill kernel assigns one program to each query head.
It avoids physical K/V duplication, but every query head in a GQA group scans
the same compact K/V tiles. This experiment asks whether assigning multiple
query heads to one program can reuse those tiles and close the gap to
FlashAttention-class prefill.

For `R` query heads per program, grouped query tile `Tg`, and Q-head-owned
reference tile `Tref`, the nominal scan reduction is:

```text
effective_kv_reuse = R * Tg / Tref
```

A value above one means fewer complete K/V scans than the reference. It does
not include register pressure, occupancy, synchronization, or softmax cost.

## Implementation

`stream_attention/kernels/grouped_gqa_prefill_triton.py` flattens the query
head and query-row dimensions into one program row axis:

```text
program rows = R * TILE_M
```

Each flattened row retains its own exact streaming online-softmax state:

```text
running maximum
running denominator
FP32 output numerator
```

The program loads each compact K/V tile once and applies it to all `R` query
heads. BF16 Q/K and probability/V operands remain BF16 for tensor-core dot
products while all online-softmax state and dot-product accumulation remain
FP32. No attention matrix or repeated K/V tensor is materialized.

The floor is deliberately separate from `stream_attn.prefill()`. It cannot be
selected by the production planner unless a measured cell beats the promoted
baseline.

## Correctness

The final gate passed on one H100 and one B200. Each architecture ran eight
native cases spanning:

```text
R = 2, 4
D = 64, 128
dtype = FP32, BF16
B1, S128, Hq8/Hkv2, causal
```

Output and log-sum-exp were checked against exact PyTorch SDPA/reference math.
The complete test file reported `9 passed` on each GPU, including the CPU reuse
contract test. Maximum BF16 output error in the measured phase was `0.0078125`.

## Performance Phase

`benchmarks/profile_grouped_gqa_prefill_floor.py` measures the grouped floor,
the first StreamAttn compact-GQA forward, and forced Flash SDPA GQA in the same
process. BF16 cells cover B1/B2, S128/S512/S1024, D64/G4 and D128/G8.

| GPU / shape | Best speedup vs first native forward | Best paired speed vs Flash SDPA | Decision |
|---|---:|---:|---|
| H100, D64/G4 | `1.41x` | `0.75x` | Not promoted |
| B200, D64/G4 | `1.52x` | `0.75x` | Not promoted |
| H100, D128/G8 | `4.85x` | `0.83x` | Not promoted |
| B200, D128/G8 | `2.72x` | `0.89x` | Not promoted |

The B200 D128 frontier was additionally swept over `TILE_N=32/64/128`, four
and eight warps, and one to three pipeline stages. No configuration crossed
`1.0x` versus Flash SDPA. `R8 x TILE_M32` with `TILE_N=128` also exposed a
262,152-byte shared-memory requirement, beyond B200's 232,448-byte limit.

## Interpretation

The experiment validates grouped exact online-softmax math and substantially
improves StreamAttn's first native GQA forward. It rejects a stronger claim:
K/V reuse by itself is not enough to beat FlashAttention.

Several low nominal-reuse schedules beat higher-reuse schedules. The governing
objective is therefore:

```text
net grouped gain =
    saved K/V scans
  + better tensor-core row geometry
  - accumulator/register pressure
  - lost occupancy
  - online-softmax rescaling cost
```

The generic Triton program cannot independently schedule asynchronous K/V
movement, tensor-core work, and softmax/rescaling at the level required to
close the final gap. More Triton tile tuning is low-value after this phase.

## Next Boundary

The next forward experiment must preserve the same grouped mathematical
mapping but change the hardware pipeline:

```text
H100: WGMMA + staged/TMA K/V movement + warp-specialized softmax
B200: tcgen05 MMA + TMEM accumulators + Blackwell-specific rescaling pipeline
```

Backward should not be built yet. First require exact forward parity or a win
against Flash SDPA in at least one meaningful H100 or B200 cell. If forward
crosses that gate, split backward into query-owned dQ and KV-group-owned dK/dV
reduction to remove global atomic contention.
