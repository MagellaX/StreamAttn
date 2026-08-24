# Selected Paged Route ABI

## Purpose

StreamAttn already has two independently useful pieces:

```text
verified reduced-work tile selection
native exact paged online-softmax kernels
```

The missing bridge is a page-native selected executor. The route ABI makes
that bridge explicit without creating a gathered K/V buffer.

## Semantic Schedule

`AttentionRouteCSR` is the canonical irregular schedule:

```text
row_ptr:  int32 [rows + 1]
atom_ids: int32 [nnz]
```

Atoms are logical. They are not physical page IDs and never contain K/V
values. A route row has one of three owners:

```text
batch     -> one selection shared by every head in a request
kv_group  -> one selection per (batch, KV head)
q_head    -> one selection per (batch, Q head)
```

The schedule also carries an ABI version and monotonically increasing epoch.
Optional per-atom head masks are legal for KV-group rows. Q-head rows encode
head ownership in their row index.

## Physical Lowering

For a page-16 cache, four physical page atoms form one 64-token compute route:

```text
page 0 -> shared rows  0..15
page 1 -> shared rows 16..31
page 2 -> shared rows 32..47
page 3 -> shared rows 48..63
```

`PackedPagedRoute64` stores:

```text
row_ptr                 int32 [B * Hkv + 1]
logical_atom_origins    int32 [routes, 4]
physical_page_ids       int32 [routes, 4]
atom_valid_masks        int32 [routes]
active_head_masks       int32 [routes, 4]
token_valid_masks       int32 [routes, 4]
route_flags             int32 [routes]
```

Logical origins remain separate from physical locators because softmax/mask
semantics and page loads answer different questions. Per-atom head masks are
necessary: two fragments packed into one route may have different Q-head
owners after a Q-head-private selector is unioned for a group-shared kernel.

The implementation is metadata-only. K/V tensor storage is neither copied nor
repacked.

## GQA Economics

For Q-head selections `S_h` inside one GQA group, group-shared execution loads
their union. Its useful-work ratio is:

```text
E_group = sum_h |S_h| / (G * |union_h S_h|)
```

Interpretation:

```text
E_group near 1 -> heads agree; group-shared WGMMA is attractive
E_group low    -> union traffic is inflated; head-private or exact may win
```

This value is emitted by physical lowering and belongs in backend selection.
It is not a safety score.

## Mutability Contract

Prepared routes cache physical page IDs. They therefore record:

```text
schedule epoch
page-table storage pointer and tensor version
sequence-length storage pointer and tensor version
```

Changing a logical schedule, remapping the live page table, or changing a
ragged length invalidates the prepared route. The runtime must re-prepare
metadata or reject execution. This prevents stale CUDA-graph-style replay.

## Scheduler Policy

The route ABI does not force one scheduler:

```text
uniform route counts -> direct static grid
moderate raggedness  -> compact static task list
high variance        -> persistent queue, only after profiling
```

Persistent scheduling is deliberately not the first implementation. NVIDIA's
[grouped scheduler documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/grouped_scheduler.md)
shows that persistent workers reduce launch/prologue costs but add scheduler
work. That trade is favorable only when the workload needs it. Blackwell
[Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
is a later SM100 lowering of the same ABI, not a separate semantic system.

FlashInfer's current block-sparse interfaces also separate sparse row
structure from a planned execution object. StreamAttn's distinction is that
the same route contract feeds its own exact fallback, distribution verifier,
and architecture-specific online-softmax backends.

## Promotion Gate

The H100 executor must be measured first with precomputed routes so selector
cost does not obscure backend cost:

```text
T_selected = T_prepare / reuse
           + T_dispatch
           + T_page_metadata
           + T_selected_mainloop
           + T_merge
           + p_verify * T_canary
```

Promotion requires total selected runtime at least 10% below exact-native,
paired wins, no K/V gather, live page-remap correctness, and the existing
closed-loop distribution gate. The 10% margin is an engineering guard against
selector and distribution variance, not an attention identity.

## H100 Static Executor

The first executor now consumes this ABI directly. Its static schedule is:

```text
grid = B * Hkv * max_routes_per_row
one producer CTA = one PackedRoute64 record
workspace = [B * Hkv, max_routes_per_row, 8, D]
```

Rows with fewer routes emit neutral partials (`O=0`, `LSE=-inf`). Per-atom
head and token masks are applied to QK scores before online softmax. Existing
exact split-state merge math then combines route partials. This preserves the
exact result over each Q head's selected token set, including Q-head-private
routes unioned into a group-shared WGMMA launch.

The measured 32K D128/G8 phase establishes two dispatch regions:

```text
selected tokens <= 16K -> static selected executor won measured B1/B4/B8 cells
selected tokens == 32K -> exact split scheduler wins at B4/B8
```

The first general lowering takes roughly 2-5 ms after framework warmup. It is
therefore suitable for a reused fixed schedule, not a per-token query-dynamic
route. Dynamic selection requires a dedicated GPU preparation kernel before
promotion. See [the complete evidence](paged_selected_h100_phase_20260825.md).
