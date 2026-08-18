# H100 exact-native D128/G4 phase diagram (2026-08-19)

## Claim

StreamAttn now has a promoted exact decode region for:

```text
GPU:       NVIDIA H100 80GB HBM3 (SM90)
dtype:     BF16
head dim:  128
Q heads:   32
KV heads:  8
GQA group: 4
KV layout: contiguous [B,Hkv,N,D]
semantics: full all-token exact attention
```

These cells use StreamAttn's native transposed true-GQA WGMMA producer and
one-warp online-softmax-state merge. FlashInfer is only the paired exact
benchmark reference; it is not called by the serving route.

## D128 kernel change

D128 doubles the QK inner dimension from four to eight `m64n8k16` WGMMA
steps and requires two PV M-tiles. A naive two-stage K/V pipeline exceeded the
SM90 per-block shared-memory limit. The working specialization therefore:

1. keeps two K stages;
2. aliases V onto the K stage after QK has consumed it;
3. prefetches the next K stage before QK;
4. loads current V while online softmax updates run; and
5. selects the active shared-memory tensor descriptor before WGMMA, keeping
   QK/PV WGMMA outside the runtime stage branch.

The last change addresses a ptxas diagnostic that inserted warp-group arrival
barriers and serialized WGMMA inside divergent control flow. D64 keeps its
existing separate K/V pipeline.

## Phase evidence

The original shared-lifetime D128/G4 pipeline produced 7/15 raw paired wins.
Predicated stage descriptors expanded that to 10/15 in the first full phase
diagram. A separate higher-repetition confirmation produced 8/9 raw wins for
`B >= 4`; every winning cell won all 15 paired trials.

The public `StreamAttnExactNativeDirectRunner` gate then rejected two fragile
cells that a raw-only gate would have admitted:

```text
B8/32K:  raw ratio 0.99971x in the third run
B16/16K: serving ratio 0.99961x
```

They are deliberately absent from the registry.

## Promoted serving cells

| Batch | KV length | Splits | Producer CTAs | Serving median | Serving minimum | Paired wins |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 32K | 8 | 256 | 1.00528x | 1.00452x | 9/9 |
| 4 | 64K | 8 | 256 | 1.00192x | 1.00109x | 9/9 |
| 8 | 16K | 4 | 256 | 1.00789x | 1.00745x | 9/9 |
| 8 | 64K | 4 | 256 | 1.01172x | 1.01027x | 9/9 |
| 16 | 32K | 2 | 256 | 1.00981x | 1.00877x | 9/9 |
| 16 | 64K | 2 | 256 | 1.00782x | 1.00736x | 9/9 |

All cells passed exact-reference checks, deterministic replay, finite-output
checks, and live-query mutation checks. The planner compiles and allocates
once; timed serving reuses fixed buffers.

## Negative controls

The negative cells are part of the result:

```text
D128/G4 B4/16K:  0.99423x, 0/9 paired wins
D128/G4 B8/32K:  0.99971x, 2/9 raw paired wins in the serving run
D128/G4 B16/16K: 0.99961x public-serving median
D128/G8 B4/32K:  0.98980x, 0/15 paired wins
```

The G8 control uses all eight physical WGMMA columns and still loses. The D128
win is therefore not explained by tensor-core column utilization. G4 doubles
the number of independent KV groups relative to G8, and every promoted cell
selects exactly 256 producer CTAs. The current edge is the interaction between
true-GQA transposition, producer parallelism, and stage scheduling.

## Relation to established kernels

The code-level comparison explains the remaining gap:

- FlashInfer's decode kernel interleaves staged `cp.async` K/V loads with QK,
  state update, and PV in one cooperative block.
- FlashAttention's Hopper forward kernel separates a low-register producer
  warp group from high-register MMA consumers with TMA pipelines.
- StreamAttn D128 currently uses cooperative `cp.async` plus transposed GQA.

The predicated-stage experiment removed one local serialization source, but
the remaining 0.2-1.2% serving margin is narrow. The next high-value exact
backend experiment is a TMA producer/consumer D128 specialization. More split
sweeps are unlikely to change the architecture: all useful cells already
converge on the same 256-CTA producer floor.

## Evidence artifacts

```text
artifacts/gate0/sm90_exact_d128_g4_phase_diagram_h100_20260819.json
artifacts/gate0/sm90_exact_d128_g4_predicated_stage_phase_h100_20260819.json
artifacts/gate0/sm90_exact_d128_g4_predicated_stage_confirm_h100_20260819.json
artifacts/gate0/sm90_exact_d128_g4_promoted_serving_gate_h100_20260819.json
artifacts/gate0/sm90_exact_d128_g8_predicated_stage_control_h100_20260819.json
```

## Boundary

This is an exact-kernel victory over FlashInfer for six measured H100 cells,
not a universal D128 claim. Unlisted shapes fail closed to another exact-native
runner or remain unsupported; they do not silently use this specialization.
