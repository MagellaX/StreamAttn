# Native SM100 Exact GQA Prefill

## Result

StreamAttn now has a promoted Blackwell-native exact causal prefill path for a
narrow BF16 GQA phase:

```text
GPU:        NVIDIA B200 / SM100a
layout:     contiguous BSHD
shape:      Hq=16, Hkv=2, group=8, D=128
semantics:  exact self-causal prefill
tile:       8 query heads x 2 query positions
```

The public `stream_attn.prefill(...)` planner selects this backend only for:

```text
B1: S64, S128, S256, S384
B2: S64
```

Every other shape or feature continues through the existing exact planner.

## Architecture

The backend extends the native SM100 TGV GQA pipeline rather than wrapping an
external attention implementation. One CTA owns one KV head, eight associated
query heads, and one or more query positions. It uses:

- TMA for Q/K/V movement;
- `tcgen05` MMA for QK and PV;
- TMEM for FP32 accumulators;
- streaming online softmax across 128-token KV tiles;
- query-tile causal truncation to avoid scheduling future KV tiles;
- row-specific masking within the diagonal KV tile;
- direct BSHD output with no GQA K/V expansion or layout repacking.

For a query tile ending at position `e`, the scheduler scans only:

```text
KV tiles = ceil(e / 128)
```

rather than `ceil(S / 128)`. The row mask then removes tokens after each
individual query position in the final tile. This preserves exact causal
semantics while approaching triangular rather than square QK/PV work.

## Resource Boundary

The original TGV overlap policy allows only half of TMEM for the current kernel.
For `R = CTA_query_heads * CTA_query_positions`, each QK and PV accumulator
reserves `max(R, 32)` TMEM columns. The legal condition is:

```text
2 * max(R, 32) < 128
```

Therefore `R <= 32`. The compiled autotune set is:

```text
h8_q1: R=8
h8_q2: R=16
h8_q4: R=32
```

`h8_q8` and `h8_q16` are structurally illegal under this overlap policy.
`h8_q4` also exceeds the dynamic shared-memory limit with three K and three V
stages, so it uses a two-stage/two-stage pipeline. The measured winner is
`h8_q2`.

## Correctness

The B200 gate covered all three legal tile variants at B1/S128, every promoted
cell, and the public planner route. The final run passed `11/11` GPU tests.
Outputs were compared with exact causal SDPA; maximum BF16 absolute error across
the promoted phase was `0.00390625`.

## Performance Method

Reference: forced PyTorch Flash SDPA GQA, PyTorch `2.7.1+cu128`.

The Flash backend context is established outside the timed loop. Before timing,
an 8192-square BF16 tensor-core workload stabilizes clocks. Each reported cell
is the median of five alternating paired trials; each trial uses 20 warmups and
100 timed calls. The StreamAttn plan reuses its output and sequence-length
buffers.

| Batch | Sequence | StreamAttn ms | Flash ms | Median paired speedup | Decision |
|---:|---:|---:|---:|---:|---|
| 1 | 64 | 0.00619 | 0.01020 | 1.647x | Promote |
| 1 | 128 | 0.00619 | 0.01072 | 1.733x | Promote |
| 1 | 256 | 0.01127 | 0.01384 | 1.221x | Promote |
| 1 | 384 | 0.01642 | 0.01853 | 1.128x | Promote |
| 1 | 512 | 0.02257 | 0.02114 | 0.936x | Fallback |
| 2 | 64 | 0.00618 | 0.01037 | 1.676x | Promote |
| 2 | 128 | 0.01028 | 0.01003 | 0.976x | Fallback |
| 2 | 256 | 0.01851 | 0.01853 | 1.001x | Fallback; no margin |
| 2 | 384 | 0.02872 | 0.02665 | 0.928x | Fallback |
| 2 | 512 | 0.03880 | 0.02260 | 0.582x | Fallback |

The benchmark is `benchmarks/profile_sm100_gqa_prefill.py`.

## H100 Boundary

The existing H100 exact WGMMA backend is intentionally not reused for this
route. It is decode-shaped: an `m64n8` WGMMA maps eight GQA heads for one query
token, writes split FP32 state, and launches a separate merge kernel. Flattening
prefill query positions through that path would multiply producer and merge
work by sequence length and preserve the wrong scheduling geometry.

The generic KV-group-owned Triton floor remains correct on H100 but below Flash
SDPA. A competitive Hopper prefill backend therefore needs a separate
multi-query-row WGMMA forward schedule with a direct epilogue, not a disguised
decode loop.

## Scope

This is an exact-kernel result, not a reduced-token or model-policy result. It
does not claim universal prefill superiority. It establishes that StreamAttn's
native engine can beat a FlashAttention-class reference in guarded Blackwell
prefill cells while preserving full attention and exact causal online-softmax
semantics.
