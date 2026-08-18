# SM90 D128 pipeline ablation (2026-08-19)

## Question

Can an architecture-valid TMA or warp-specialized mainloop turn the promoted
D128/G4 exact-native region from a narrow FlashInfer win into a durable 2-4%
kernel family?

The measured target was:

```text
GPU:       NVIDIA H100 80GB HBM3 (SM90)
dtype:     BF16
batch:     4
Q heads:   32
KV heads:  8
GQA group: 4
KV length: 32768
head dim:  128
```

These are exact full-context QK+PV probes. FlashInfer is not called by the
experimental kernels.

## Architecture correction

A 160-thread CTA containing one 32-thread producer warp followed by one
128-thread WGMMA consumer warpgroup is not a valid Hopper WGMMA topology.
Warpgroup operations require the four consumer warps to form an aligned
128-thread warpgroup. The valid prototype therefore uses:

```text
threads 0..127:   producer warpgroup
threads 128..255: aligned WGMMA consumer warpgroup
```

All producer threads participate in pipeline barriers, but only the producer
warpgroup performs copy work. The producer warpgroup deallocates registers and
the consumer warpgroup receives a configurable register budget.

The measured independent-buffer topology is `2K + 1V`. Its actual compiled
dynamic shared-memory allocation is 52,864 bytes for the QK+PV probe. The
isolated TMA copy probe uses 49,792 bytes for its `2K + 1V` storage and barrier
state. Opt-in dynamic shared memory is required.

## TMA data-movement floor

The first experiment removed WGMMA and softmax from the question. It compared
cooperative `cp.async` with a single elected TMA issuer for K-only and K+V
traffic, using the same total logical 32K/B4/G4 tile workload.

The table reports `cp.async time / TMA time`; values below one mean TMA lost.

| Producer CTAs | Tiles/CTA | K-only | K+V |
|---:|---:|---:|---:|
| 512 | 32 | 0.9071x | 0.9374x |
| 256 | 64 | 0.8224x | 0.8554x |
| 128 | 128 | 0.7032x | 0.7591x |

The result is stable in eager execution and CUDA-graph device-floor timing.
All checksums matched exactly.

Resource measurements explain part of the trade:

| Probe | Registers/thread | Dynamic shared | Blocks/SM |
|---|---:|---:|---:|
| cooperative K | 34 | 49,664 B | 4 |
| cooperative K+V | 56 | 49,664 B | 4 |
| TMA K | 32 | 33,408 B | 6 |
| TMA `2K + 1V` | 40 | 49,792 B | 4 |

TMA improves the K-only resource floor, but the elected-thread TMA and
transaction-barrier path is slower for every measured CTA duration. This is a
device-side result; CUDA graph replay does not repair it.

## Warp-specialized cp.async QK+PV floor

The second experiment retained the transposed WGMMA QK+PV operations and
isolated role specialization while keeping `cp.async` as the copy primitive.
It compared:

```text
cooperative baseline:
  128 threads
  cooperative cp.async
  transposed QK + PV WGMMA

warp-specialized candidate:
  256 threads
  producer warpgroup + aligned consumer warpgroup
  2K + 1V shared-memory pipeline
  transposed QK + PV WGMMA
```

An initial 160-register consumer allocation allowed only one block/SM and
produced a 0.7401x result. A compile-time register sweep restored two blocks/SM
for 96, 112, and 128 registers.

The full five-repeat CUDA-graph device-floor results are below. Values are
`cooperative time / warp-specialized time`; values below one mean the
warp-specialized kernel lost.

| Splits | Producer CTAs | Tiles/CTA | 96 regs | 112 regs | 128 regs |
|---:|---:|---:|---:|---:|---:|
| 4 | 128 | 128 | 0.7963x | 0.8079x | 0.8062x |
| 8 | 256 | 64 | 0.9633x | 0.9643x | 0.9640x |
| 16 | 512 | 32 | 0.9704x | 0.9713x | 0.9714x |

The best result remains 2.86% slower than the cooperative baseline. Longer
CTAs are substantially worse, which shows that the fixed producer warpgroup
and per-tile pipeline/barrier protocol are not amortized by more mainloop work.
The 96-128 register variants all compile to their requested register counts,
all allow two resident blocks/SM, and all produce nearly identical latency.
Consumer register pressure is therefore no longer the limiting variable.

The checksum difference from the cooperative reference is at most 0.01172
absolute and 2.04e-4 relative across the sweep. The difference comes from the
changed accumulation order; no non-finite or replay failures were observed.

## Decision

Do not implement the full TMA warp-specialized D128 mainloop in this topology.

Both prerequisite hypotheses are negative:

```text
TMA copy floor:              slower for every CTA duration
warp-specialized cp.async:   slower after occupancy is repaired
```

Combining TMA transaction barriers with the already-negative 256-thread
producer/consumer topology is unlikely to create the required 3% anchor win.
The experiment therefore follows the predefined stop rule instead of forcing a
large rewrite.

The promoted cooperative D128/G4 exact backend remains the serving path. The
TMA and warp-specialized kernels remain isolated benchmark probes and are not
registered for dispatch.

## Next exact-backend branch

The evidence narrows the next high-risk exact experiment to work that does not
pay a permanently idle 128-thread producer warpgroup:

1. pack two G4 KV groups into one full `m64n8` WGMMA tile, if a block-diagonal
   QK/PV layout can avoid duplicating K/V traffic;
2. reduce split-state merge and public-runner overhead for the narrow cells;
3. use persistent producer scheduling only for low-CTA shapes where a CTA can
   claim multiple KV-group/split tasks without adding another warpgroup.

The first item is the only branch with a plausible step-function gain. It can
recover the four unused WGMMA columns in G4, but it should begin with a QK/PV
floor probe because cross-group data layout and masking can erase the nominal
2x tensor-column utilization.

## Evidence artifacts

```text
artifacts/gate0/sm90_d128_tma_pipeline_floor_smoke_modal_h100_20260819.json
artifacts/gate0/sm90_d128_tma_pipeline_floor_modal_h100_20260819.json
artifacts/gate0/sm90_d128_ws_cp_async_floor_smoke_modal_h100_20260819.json
artifacts/gate0/sm90_d128_ws_cp_async_register_sweep_smoke_modal_h100_20260819.json
artifacts/gate0/sm90_d128_ws_cp_async_register_sweep_modal_h100_20260819.json
```

## Boundary

This rejects one concrete SM90 D128 TMA/warp-specialization topology. It does
not claim that TMA is generally ineffective, nor that FlashAttention-style
warp specialization is ineffective for larger tiles or different attention
shapes. The result is specific to this decode geometry, its small WGMMA N tile,
and a 256-thread CTA whose producer role consumes an entire warpgroup.
