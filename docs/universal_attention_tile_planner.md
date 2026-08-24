# Universal Attention Tile Planner

## Objective

StreamAttn is one semantic attention engine, not one kernel and not one sparse
policy. Every supported request must have a StreamAttn-owned correct path. The
planner may then choose a faster specialized schedule when its semantic and
performance gates pass.

```text
exact request       -> schedule every required logical KV tile
verified request    -> schedule a validated subset of logical KV tiles
unknown request     -> exact-native
```

FlashInfer and FlashAttention remain comparison baselines. They are not the
fallback implementation of this contract.

## Planning Layers

`AttentionProblem` records semantics independently from any CUDA kernel:

```text
phase and guarantee
batch/query/KV lengths
Q and KV head geometry
head dimension and dtype
contiguous or paged cache
HND/NHD layout and physical page size
```

`AttentionTilePlan` compiles that problem into a logical KV source and a tile
schedule. Exact schedules represent all tiles without materializing a large ID
array. Reduced-work schedules carry explicit per-row logical tile IDs.

`AttentionBackendPlan` records the physical execution decision:

```text
architecture
backend variant
context split count
workspace bytes
selection reason
```

The engine exposes all three through `StreamAttnEnginePlan.summary()` so route
telemetry can distinguish semantic work reduction from backend speed.

## Current Lowering

The first integrated phase covers:

| Cache | Schedule | Guarantee | Runtime |
|---|---|---|---|
| Contiguous NHD/HND | All logical tiles | Exact | Native exact or exact reference |
| Paged NHD/HND | All logical tiles | Exact | Native paged exact or exact reference |
| Contiguous NHD | Explicit calibrated blocks | Distribution verified | Native fixed-block kernel |

Paged page-16 sources describe how four physical fragments form one logical
64-token tile. Ragged requests preserve one logical tile count per row. The
semantic plan therefore remains the same whether a backend uses Triton, SM90
WGMMA, SM80 `cp.async`, or SM100 TMA/TMEM/`tcgen05`.

## Remaining Work

The planner contract is now shared; execution is not fully unified yet. The
next engine work is:

1. Make paged selected schedules consume explicit tile IDs without gathering
   or repacking K/V.
2. Add sliding-window and query-selected schedule builders.
3. Move split/task scheduling behind the backend-plan boundary.
4. Add a live verifier that can replace selected row/layer work with exact
   tile schedules.
5. Reuse the same contract for chunked prefill and mixed prefill/decode.

Only after this H100 decode runtime is coherent should architecture expansion
be treated as the main project direction. A100 and B200 backends are important
implementations of the engine, not substitutes for it.
