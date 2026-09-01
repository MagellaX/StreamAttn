# SM90 Grouped-Prefill RS-PV Execution-State Study

## Decision

This study closes the producer-heavy Hopper grouped-prefill branch while
retaining one useful dataflow result:

```text
retain: RS-WGMMA for PV
retain: TMA multicast as a transport primitive
reject: one-producer / one-consumer TMA attention epoch
reject: one-producer / two-consumer same-CTA epoch
reject: two-CTA multicast full attention epoch
```

No complete prefill kernel is promoted by this work. The next H100 experiment
is a lean, 128-thread, consumer-owned `cp.async` attention canary that replaces
SS-PV with RS-PV. It deliberately avoids a dedicated producer warpgroup and
the staged state that reduced residency in every TMA attention floor.

## Question

The earlier natural-WGMMA canary proved exact grouped GQA geometry but reached
only `0.352x-0.596x` of graph-captured Flash SDPA. The proposed repair was a
FlashAttention-3-class state machine:

```text
TMA producer + true softmax + SS-QK + RS-PV + asynchronous consumers
```

Rather than build another complete kernel, this study decomposed that proposal
into component and execution-state floors on an NVIDIA H100 80GB HBM3.

The measured geometry is BF16 `M64 x N64 x D128`. Every floor writes a checksum
to prevent dead-code elimination. Paired implementations use identical inputs,
and short rows are checked against an FP32 reference.

## RS-PV Component Result

The first experiment compared shared/shared PV with register/shared PV, then
placed each dataflow in a serial QK, true-softmax, PV epoch.

| Measurement | RS speedup over SS |
|---|---:|
| Isolated PV | `1.0061x-1.0168x` |
| Serial attention epoch | `1.1775x-1.4938x` |

The serial RS epoch uses 123 registers per thread, 49,792 bytes of dynamic
shared memory, no local memory, and sustains four CTAs per SM. The SS epoch uses
125 registers, 57,856 bytes of shared memory, and three CTAs per SM. RS-PV
therefore removes a real probability-materialization cost and improves the
complete serial dataflow. This is a component and epoch result, not an SDPA
comparison.

Separately launched QK, softmax, and PV times are diagnostic component sums.
They are not a theoretical overlap lower bound, because the kernels have
different residency, launch structure, and state lifetimes.

## Producer Topology Matrix

All ratios below use fair vectorized serial RS work as the denominator. Values
below one are losses.

| Topology | Reuse depth | Measured ratio | Resource result | Decision |
|---|---:|---:|---|---|
| 1 producer + 1 consumer CTA | 1/2/4/8 | `0.8101x-0.9616x` | 255 registers, 82,560 B shared, 1 CTA/SM | Reject |
| 1 producer + 2 consumers, same CTA | 1 | `0.7227x` | 32 local B/thread, 1 CTA/SM | Reject immediately |
| 2-CTA multicast transport only | 1/2/4/8 | `0.9482x-0.9777x` vs independent TMA | exact, no spills | Retain primitive |
| 2-CTA multicast attention epoch | 1/2/4/8 | `0.5216x-0.7121x` vs serial | 255 registers, 82,560 B shared, 1 CTA/SM | Reject |

The fair 1+1 comparison corrected an earlier scalar/strided baseline that made
TMA appear `2.4x-3.6x` faster. Once the serial baseline used vectorized
`cp.async`, the TMA epoch lost every cell.

The same-CTA two-consumer design used the maximal legal Hopper register split:
24 registers for the producer warpgroup and 240 for each consumer warpgroup.
It still spilled and slowed down, so a broad sweep was not justified.

The cluster transport floor is positive evidence: a two-CTA cluster can
multicast K/V exactly with only a `2.2%-5.2%` throughput cost relative to two
independent TMA loads. But inserting QK, true softmax, and RS-PV exposes the
same producer/state residency problem. The complete cluster epoch remains
`28.8%-47.8%` slower than two lean serial RS CTAs.

## Root Cause

The rejected topologies all replace a four-CTA/SM serial consumer with a
one-CTA/SM producer-consumer state machine:

```text
lean serial RS:
  128 threads
  124 registers/thread
  49,792 B shared
  4 CTAs/SM

TMA attention:
  256 threads
  255 registers/thread
  82,560 B shared
  1 CTA/SM
```

TMA and multicast reduce or share K/V movement, but K/V transport is not the
exposed critical path in this geometry. The dedicated producer, circular K/V
state, barriers, and larger CTA remove enough residency that the load saving
cannot recover throughput. Greater reuse depth improves the ratios but never
crosses over.

## Compiler Evidence

The exact compiler now distinguishes `tma_multicast` from ordinary `tma`.
This matters because the transport primitive passed while the attention
topology failed. Treating both as one load engine would erase that distinction.

The durable decisions are:

```text
SM90 + SS-QK + SS-PV + symmetric consumers:
  rejected_performance

SM90 + TMA producer + SS-QK + RS-PV:
  rejected_resource_and_performance

SM90 + TMA multicast transport:
  viable_primitive

SM90 + TMA multicast + full attention epoch:
  rejected_performance

SM90 + consumer-owned cp.async + SS-QK + RS-PV:
  next_candidate
```

## Next Gate

Integrate RS-PV into the existing complete 128-thread H100 prefill canary while
keeping consumer-owned vectorized `cp.async` and the current exact online
softmax. Measure G4/G8, D128, B1, S2K/S4K against graph-captured Flash SDPA.

```text
correctness: exact output/LSE tolerances pass
resource: zero local-memory spills
canary: >= 0.90x Flash SDPA before broad tuning
promotion: > 1.00x median and >= 1.00x minimum paired ratio
```

If that canary does not reach `0.90x`, close this H100 grouped-prefill family.
Do not add another producer warpgroup, stage-count sweep, or cluster variant.

## Artifacts

```text
artifacts/gate0/sm90_grouped_prefill_epoch_floor_modal_h100_20260902.json
artifacts/gate0/sm90_grouped_prefill_tma_epoch_floor_modal_h100_20260902.json
artifacts/gate0/sm90_grouped_prefill_dual_consumer_floor_smoke_modal_h100_20260902.json
artifacts/gate0/sm90_grouped_prefill_cluster_floor_modal_h100_20260902.json
artifacts/gate0/sm90_grouped_prefill_cluster_epoch_floor_modal_h100_20260902.json
```
