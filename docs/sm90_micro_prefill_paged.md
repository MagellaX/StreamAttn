# Direct Paged and Ragged Micro-Prefill

StreamAttn's experimental SM90 micro-prefill plans can read HND or NHD
page-16 KV storage directly. This extends the retained transposed and natural
R64 families; it does not replace the promoted paged-decode implementation.

## Contract

| Input | Shape or rule |
| --- | --- |
| Query and output | `[B, Mmax, Hq, D]`, contiguous FP16/BF16 |
| Query lengths | `[B]`, contiguous int32; each value in `[0, Mmax]` |
| HND pages | `[physical_pages, Hkv, 16, D]` |
| NHD pages | `[physical_pages, 16, Hkv, D]` |
| Page table | `[B, max_pages]`, contiguous int32; active IDs in bounds |
| KV lengths | `[B]`, contiguous int32; each value in `[0, max_pages * 16]` |
| Logical positions, when causal | int64 `[B, Mmax]` and `[B, max_pages * 16]` |
| Native family domain | SM90, `Mmax=2..64`, `D=64/128`, `G=Hq/Hkv=4/8` |

All buffers share the query device. Ragged query and KV lengths are independent;
a request can have zero queries, zero KV, or no visible keys under the causal
mask. Such output rows are zero, with LSE `-inf`. Inactive page-table entries can
be `-1`; padding in the query and physical KV pages need not be initialized.
Shared physical prefix pages are supported for read-only attention.

Visibility is defined by logical positions, not by physical page order:

```text
valid(b,i,j) = i < query_lengths[b]
             and j < kv_lengths[b]
             and (not causal or key_positions[b,j] <= query_positions[b,i])
```

The kernel does not infer a top-left or bottom-right alignment. Callers supply
append positions explicitly. Reordered key positions and origins above int32
range have the same contract.

## Implementation

One shared loader maps logical tokens to physical pages and issues 16-byte
`cp.async` transactions into the existing swizzled shared-memory layouts.
It serves K for both families and the natural family's transposed V layout.
The MMA shapes, online softmax recurrence, and associative split merge remain
the retained implementations. No dense KV buffer or full attention matrix is
materialized by native execution.

Invalid tail tokens use zero-filling copies and never dereference an inactive
page-table entry. This is more than a score-mask detail: `0 * NaN` is still NaN,
so poisoned V padding must not enter PV even when its probability is zero.
The implementation uses CUTLASS's
[predicated zero-fill copy primitive](https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cute/arch/copy_sm80.hpp).

Ragged splits may be empty even when the planned capacity has many tiles.
Those CTAs publish the merge identity `(output=0, LSE=-inf)` before any K/V
preload. The natural family's balanced boundaries use 64-bit intermediate
arithmetic. Neither family changes softmax normalization or drops valid tokens.

## Planned Execution

```python
from stream_attention.paged import PagedKVCache
from stream_attention.backends.sm90.micro_prefill_paged import PagedMicroPrefillPlan

cache = PagedKVCache(k_pages, v_pages, page_table, kv_lengths, layout="NHD")
plan = PagedMicroPrefillPlan.build(
    q, cache, query_lengths,
    natural=False,  # explicit physical family, not an oracle dispatcher
    causal=True,
    query_positions=q_positions,
    key_positions=k_positions,
)
out = plan.run()
```

Planning validates metadata values, compiles the specialization and allocates
output/partial-state buffers. Replay allocates no tensors, performs no host
metadata readback and uses the current device stream. Page tables, lengths,
positions and tensor contents can be changed in place between CUDA graph
replays, within the original capacities. The caller must keep active IDs and
lengths valid and synchronize those updates. Replacing buffer pointers requires
a new plan and graph.

`PagedKVCache.validate_prefill` shares the cache validator with decode, while
leaving decode's single-query/positive-KV contract unchanged. The native host
entry point additionally checks tensor geometry, dtype, device, workspace
shape, launch limits and output/input overlap.

## Verification Protocol

The full matrix contains 144 cases spanning both layouts, both dtypes, D64/D128,
G4/G8, B1/B2/B4, query capacities 2/3/9/17/32/64 and KV capacities from 16 to
16,400 tokens. It includes page capacities not divisible by four, non-dividing
splits, shared prefixes, poisoned padding, arbitrary physical page order,
noncausal attention and explicit-position causal masks.

Each family is checked against an independent FP32 output/LSE reference in
three states: eager execution, in-place mutated graph replay and all-empty KV
graph replay. Reference-only gathering is outside native execution and timing.
The harness validates each mutated fixture before replay. Independent runs use
different random seeds.

The initial smoke artifact is **invalidated**, not counted as a success. Its B4
fixture could mutate a query length to `Mmax + 1`. The broader initial runs were
stopped, the fixture was bounded and a regression test added. See the
[invalidation record](../artifacts/gate0/sm90_micro_paged_initial_run_invalidation_20260905.json).
Accepted evidence uses the corrected `_v2` artifact names.

## H100 Results

| Run | Passed cases | Output/LSE checks | Max BF16 output error | Max FP16 output error | Max LSE error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lightning, seed 7013 | 144/144 | 864 | 0.0078562 | 0.0010720 | 0.000006676 |
| Modal independent replay, seed 9613 | 48/48 | 288 | 0.0080221 | 0.0009945 | 0.000004769 |

Each check compares both output and LSE; empty-row identities are also checked
explicitly. Both runs used PyTorch 2.7.1, CUDA 12.8 and H100 80GB HBM3. The
Lightning job reported 1.0302889 credits and was deleted after result retrieval.
The earlier invalidated Lightning run cost 0.3329111 credits; it is not counted
as successful evidence.

Raw results and normalized source hashes:

- [Lightning full matrix](../artifacts/gate0/sm90_micro_paged_lightning_h100_20260905_v2.json)
- [Lightning build/run log](../artifacts/gate0/sm90_micro_paged_lightning_h100_20260905_v2.log)
- [Independent Modal replay](../artifacts/gate0/sm90_micro_paged_modal_h100_20260905_v2.json)

The local suite passed 972 tests, with 71 hardware-dependent skips. Two new
H100 pytest cases exercise the poisoned-page graph fixture when SM90 and
CUTLASS headers are available. Offline CUDA CI now includes representative
D64/FP16 and D128/BF16 paged causal builds, each instantiating both layouts and
both families; runtime GPU runs covered the complete dtype/mask cross product.

This is an experimental backend, not a new public phase-database promotion.
The next integration test must time a complete mixed-ragged plan against
compatible paged baselines, including query packing and metadata preparation
where needed. The M64 producer bottleneck remains a separate, measured research
question; successful addressing and masking do not establish a performance win.
