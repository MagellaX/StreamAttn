# SM90 Consumer-Owned RS-PV Grouped Prefill

## Decision

The lean Hopper grouped-prefill branch passed its complete-kernel canary and a
30-cell H100 phase screen. StreamAttn now has an exact, compact-GQA D128 kernel
that beats graph-captured Flash SDPA in a scoped long-sequence region:

```text
architecture: SM90
layout: BSHD
dtype: BF16
attention: causal, exact
query heads: 16
GQA groups: 4 or 8
sequence length: a multiple of 64
```

This is not a universal prefill promotion. The current implementation is a
canary plan, supports equal sequence lengths, and has not yet been connected to
the public exact phase compiler.

## Kernel

The winning topology removes the producer warpgroup and lets one 128-thread
consumer warpgroup own the entire attention state:

```text
Q: consumer-loaded once per M64 tile
K: two cp.async stages, N64 tiles
QK: shared/shared WGMMA
softmax: exact causal FP32 online recurrence
P: retained in registers
V: one transposed shared tile
PV: register/shared WGMMA
output: direct BF16 write plus FP32 LSE
```

For each key tile, the kernel updates row state with

```text
m_new = max(m_old, max(scores))
alpha = exp(m_old - m_new)
l_new = alpha * l_old + sum(exp(scores - m_new))
o_new = alpha * o_old + exp(scores - m_new) @ V
```

The final output is `o_new / l_new`. This remains full exact attention; it does
not select, drop, or approximate KV tokens.

The compiled kernel uses 168 registers per thread, 65,536 bytes of dynamic
shared memory, zero local memory, and allows three CTAs per SM. Relative to the
earlier shared/shared PV canary, register/shared PV improves complete-kernel
latency by about `1.69x-2.15x` over the broad H100 screen.

## Correctness Contract

Every measured cell compares the complete output with forced Flash SDPA and
checks sampled LSE rows against an FP32 dense reference. The promotion screen
requires:

```text
max absolute output error <= 0.04
relative L2 output error <= 0.02
sampled LSE max absolute error <= 0.01
local memory per thread == 0
median paired Flash/StreamAttn ratio > 1.00
minimum paired Flash/StreamAttn ratio >= 1.00
```

All 30 H100 cells passed correctness and resource checks. Observed output max
absolute error was at most `0.0078125`; sampled LSE max absolute error was at
most `1.91e-6`.

## Independent Canary

The B1 S2K/S4K G4/G8 canary was measured in independent Hopper environments.
The named H100 replay reported:

| Group | Sequence | Median vs Flash | Minimum paired | vs prior SS-PV |
|---:|---:|---:|---:|---:|
| G4 | 2K | `1.004x` | `0.975x` | `1.914x` |
| G4 | 4K | `1.097x` | `1.085x` | `2.091x` |
| G8 | 2K | `1.068x` | `1.041x` | `2.001x` |
| G8 | 4K | `1.120x` | `1.101x` | `2.051x` |

The second Hopper allocation was an H200. It also passed every canary
correctness/resource gate and met the `0.90x` broad-tuning threshold in every
cell. Device names are taken from `torch.cuda.get_device_name()` in each raw
artifact rather than inferred from the requested provider class.

## H100 Promotion Matrix

The broad H100 screen covers batches `1/2/4`, sequence lengths
`512/1024/2048/4096/8192`, and G4/G8. Eleven of 30 cells passed the strict
paired promotion gate:

| Batch | G4 promoted sequences | G8 promoted sequences | Best median speedup |
|---:|---|---|---:|
| 1 | 4K, 8K | 2K, 4K, 8K | `1.18x` |
| 2 | 4K, 8K | 4K, 8K | `1.05x` |
| 4 | 8K | 8K | `1.05x` |

The boundary moves to longer sequences as batch grows. That is consistent with
the kernel launching an M64 CTA for each flattened query-head tile: increasing
batch exposes more parallel rows but does not reduce work per row, while Flash
SDPA uses the larger launch surface more efficiently at short and medium
sequences. The result is a phase boundary, not evidence for one global winner.

The B1/G4/S2K cell remains a boundary case. Its H100 medians were `1.000x` and
`1.004x` in the broad and independent canary runs, but its minimum paired ratios
were below one, so it is not promoted.

## Provider Evidence Boundary

The independent H100 canary completed. The first broad Lightning attempt was
invalidated because an anonymous dependency clone blocked on an interactive
credential prompt; the launcher now uses a pinned archive. Two corrected broad
attempts were then rejected during provider provisioning before a container
started. Those jobs were deleted and are not counted as benchmark evidence.

The 30-cell promotion map therefore comes from one complete H100 run. Before
public auto-routing, the 11 proposed cells should receive a targeted independent
replay on another H100. Failed provisioning is an infrastructure result, not a
kernel regression and not a substitute for replication.

## Next Gate

Do not reopen producer-heavy TMA, symmetric-consumer, or cluster-attention
tuning. The next high-signal work is:

1. Replay the 11 proposed cells on an independent H100.
2. Insert only cross-provider winners into the exact phase database.
3. Add guarded public dispatch with Flash SDPA fallback outside those cells.
4. Expand the shape surface across query-head counts, GQA groups, D64/D256,
   FP16, noncausal attention, and ragged sequences.

That sequence turns this kernel result into universal-engine coverage without
pretending the measured region is universal.

## Artifacts

```text
artifacts/gate0/sm90_grouped_rs_prefill_smoke_modal_h100_20260902.json
artifacts/gate0/sm90_grouped_rs_prefill_canary_modal_h200_20260902.json
artifacts/gate0/sm90_grouped_rs_prefill_canary_lightning_h100_20260902.json
artifacts/gate0/sm90_grouped_rs_prefill_promotion_modal_h100_20260902.json
```
