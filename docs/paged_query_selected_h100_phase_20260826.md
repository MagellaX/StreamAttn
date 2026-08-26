# Query-Selected Paged Decode on H100

Date: 2026-08-26

## Question

The previous dynamic executor proved that mutable GPU Q-head routes can be
lowered and executed faster than FlashInfer exact attention. It deliberately
excluded the selector. This phase asks the harder systems question:

```text
Can a live query choose paged KV atoms and still leave a net end-to-end win?
```

The measured path is:

```text
query
  -> support-key score
  -> fixed-width top-k Q-head atoms
  -> bounded GQA membership union
  -> live page resolution
  -> selected WGMMA online softmax
  -> exact split-state merge
```

There is no CPU route construction, route-count readback, KV gather, KV
repacking, or GPU sort in the timed loop.

The benchmark uses a fixed 32K cache. In a live growing cache, the current
recent atom is included unconditionally; when that atom becomes a selectable
middle atom, its support metadata must be finalized. The incremental
once-per-64-token metadata update is not yet part of this runtime path.

## Design

Each logical atom contains 64 KV tokens. Planning/prefill builds `P` support
vectors per atom in logical page order:

```text
support_keys: [B, Hkv, N/64, P, D]
```

The decode selector approximates the atom support function

```text
h_a(q) = max_{k in atom a} q dot k
```

with

```text
h_hat_a(q) = max_{p < P} q dot support[a,p].
```

It always includes one sink and one recent atom, then emits the four best
middle atoms for S384. Output is in score order. The downstream membership map
uses atomic bit insertion to detect duplicates and compacts active atoms in
canonical logical order, eliminating a standalone sort.

The selector dot-work ratio versus a full token QK scan is:

```text
P / 64
```

P4 scans 6.25% of the token-QK dot products. P8 scans 12.5%.

## H100 Evidence

Shape:

```text
GPU:       H100 / SM90
KV:        NHD page-16, BF16, 32K
Attention: D128, Hq16, Hkv2, G8
Route:     S384 = 6 x 64-token atoms per Q head
Baseline:  fastest supported FlashInfer 0.6.17 exact backend
```

P1/P2/P4 phase:

| B | P | Complete StreamAttn ms | FlashInfer ms | Median speedup | Paired block-timed minimum | Oracle middle recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 0.06458 | 0.03699 | 0.573x | 0.383x | 0.0% |
| 1 | 2 | 0.06619 | 0.03669 | 0.554x | 0.398x | 9.4% |
| 1 | 4 | 0.06496 | 0.03678 | 0.566x | 0.415x | 23.4% |
| 4 | 1 | 0.06728 | 0.07515 | 1.117x | 1.153x | 1.2% |
| 4 | 2 | 0.06643 | 0.07560 | 1.138x | 1.161x | 5.1% |
| 4 | 4 | 0.06694 | 0.07506 | 1.121x | 1.160x | 15.2% |
| 8 | 1 | 0.06984 | 0.11874 | 1.700x | 2.025x | 3.1% |
| 8 | 2 | 0.06944 | 0.11821 | 1.702x | 2.042x | 10.7% |
| 8 | 4 | 0.07038 | 0.11872 | 1.687x | 2.009x | 20.5% |

The paired figures time 100 calls inside each CUDA event interval and alternate
measurement order. This avoids event-level jitter overwhelming sub-0.1 ms
kernels.

P8 risk phase:

| B | Summary construction | Complete ms | FlashInfer ms | Speedup | Paired minimum | Oracle recall |
|---:|---|---:|---:|---:|---:|---:|
| 4 | centroid + extremes | 0.07741 | 0.07718 | 0.997x | 0.968x | 33.2% |
| 4 | centroid + top norm | 0.07776 | 0.07704 | 0.991x | 0.937x | 33.6% |
| 8 | centroid + extremes | 0.07722 | 0.11850 | 1.535x | 1.920x | 35.5% |
| 8 | centroid + top norm | 0.08536 | 0.11915 | 1.396x | 1.700x | 36.1% |

## Conclusions

1. Query selection can remain net-positive. This is no longer a benchmark with
   precomputed routes: B4 and B8 P1/P2/P4 beat FlashInfer after including the
   selector and compiler.
2. The selector has a fixed floor near 0.05 ms. B1 cannot amortize it and must
   use exact fallback until selection is fused or made persistent/cooperative.
3. Sketch width is a routing dimension. P4 is the measured B4 frontier; P8
   improves proxy quality but consumes all B4 margin. B8 can afford P8.
4. Top-norm representatives are not worth carrying forward. Their tiny recall
   change does not pay for the latency increase.
5. Synthetic Gaussian block-max recall is a kernel diagnostic, not a policy
   gate. Real model keys occupy different geometry, and distribution safety
   must be measured separately.

The measured cost objective is therefore:

```text
choose P, S = argmin T_select(B,N,P) + T_route(B,S) + T_attn(B,S)
subject to model_distribution_error <= epsilon.
```

Current systems frontier:

```text
B1: exact fallback
B4: P4 / S384 research candidate
B8: P8 / S384 research candidate when the stronger sketch is needed
```

## Safety Boundary and Next Research

The selected WGMMA kernel is exact over the chosen atoms. The selector is not
exact full-context attention. Existing Qwen stress analysis found that an
offline P4 extreme-support proxy with exact refinement over 32 candidates came
close to the block-QK oracle on coverage, but late-layer value/composition risk
still prevented adversarial promotion.

The next high-signal selector experiment is therefore not a larger fixed
sketch. It is a runtime two-stage path:

```text
P4 support scan -> top-32 candidate atoms -> exact block-max refinement
-> final four middle atoms -> existing no-sync executor
```

That scans about 6.25% of tokens during exact refinement in addition to the
6.25% P4 support scan. It should only be integrated if the complete B8 path
remains positive and real-model replay shows a material safety gain. Unknown or
failed request tiers remain on exact native attention.

## Artifacts

- `artifacts/h100/paged_query_selected_phase_20260826.json`
- `artifacts/h100/paged_query_selected_p8_20260826.json`
- `artifacts/h100/paged_query_selected_smoke_20260826.json`
- `docs/qwen25_3b_dynamic_selector_findings.md`
