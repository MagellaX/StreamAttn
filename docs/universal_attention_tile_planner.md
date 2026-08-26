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
array. Reduced-work schedules may carry host rows for policy inspection and an
immutable device CSR payload for execution. Route rows explicitly belong to a
batch item, a `(batch, KV head)` group, or a `(batch, Q head)` pair.

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
| Paged NHD/HND | Device CSR -> `PackedRoute64` | Distribution verified | H100 static selected WGMMA |

Paged page-16 sources describe how four physical fragments form one logical
64-token tile. Ragged requests preserve one logical tile count per row. The
semantic plan therefore remains the same whether a backend uses Triton, SM90
WGMMA, SM80 `cp.async`, or SM100 TMA/TMEM/`tcgen05`.

Selected paged schedules now lower through two additional contracts:

```text
AttentionRouteCSR
  logical atom IDs + explicit batch/KV-group/Q-head row ownership

PackedPagedRoute64
  four logical origins + four physical page IDs
  per-atom validity, token-tail mask, and active-Q-head mask
```

The physical record carries a head mask for each page atom, not one mask for
the whole 64-token route. This is required when Q heads in one GQA group select
different atoms. The lowering reports group-route efficiency:

```text
E_group = sum_h |S_h| / (G * |union_h S_h|)
```

High overlap favors group-shared WGMMA. Low overlap is a planner signal for a
head-private backend or exact-native fallback.

Prepared physical routes record the schedule epoch and the PyTorch storage
pointer/version of the page table and sequence lengths. An in-place page remap
therefore invalidates prepared metadata instead of silently reading stale
physical pages. The lowering copies metadata only; it never gathers K/V.

## H100 Execution Evidence

The first physical consumer is a static SM90 executor:

```text
PackedRoute64 record -> one producer CTA
four physical page-16 atoms -> one 64-token WGMMA tile
per-atom head/token masks -> scores masked before online softmax
fixed-stride partial states -> one-warp row merge
```

At NHD/page-16 BF16 D128/G8 and 32K, all selected routes through 16K tokens
beat FlashInfer exact in the first B1/B4/B8 phase. Independent 384/2048-token
confirmation won all 90 paired trials. The 32K control lost at B4/B8, proving
the planner needs a route-density boundary between selected-static and the
existing exact split scheduler. Full measurements are in
[the H100 selected-paged phase](paged_selected_h100_phase_20260825.md).

## Remaining Work

The planner contract is now shared; execution is not fully unified yet. The
next engine work is:

1. Benchmark compact ragged tasks against the completed static executor; add a
   persistent queue only if measured task variance pays for it.
2. Add segment/run producers for structured sink/middle/recent and sliding
   schedules while retaining CSR as the irregular device ABI.
3. Connect query-aware GPU selectors to the completed no-sync Q-head CSR route
   compiler and include selector cost in the dispatch phase diagram.
4. Add a live verifier that can replace selected row/layer work with exact
   tile schedules.
5. Reuse the same contract for chunked prefill and mixed prefill/decode.

Only after this H100 decode runtime is coherent should architecture expansion
be treated as the main project direction. A100 and B200 backends are important
implementations of the engine, not substitutes for it.
