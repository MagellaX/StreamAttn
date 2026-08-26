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

## Two-Stage Exact Candidate Refinement

The runtime now implements the proposed no-sync refinement path:

```text
P4 support scan
  -> top-C middle candidates per Q head
  -> exact block-max QK over the C x 64 candidate tokens
  -> final four middle atoms
  -> existing membership compiler and selected WGMMA executor
```

The implementation preallocates FP32 support/candidate score workspaces and
int32 candidate IDs. It performs no host readback and reads candidate keys
directly from page-16 NHD or HND storage. The selected attention result remains
exact over the final route.

For P4 and C32, the token-equivalent selector work is:

```text
support scan: P / 64 = 4 / 64 = 6.25%
exact refine: C * 64 / N = 32 * 64 / 32768 = 6.25%
```

The arithmetic work is about 12.5% of a full token QK scan, but the support
scan, candidate extraction, exact refinement, and final top-k are separate GPU
stages. Launch and synchronization structure therefore matters as much as dot
count at this scale.

Phase results, with nine paired block-timed trials per cell:

| B | Refine C | Selector ms | Paired median | Paired minimum | Wins | Synthetic oracle recall |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0 | 0.05330 | 1.838x | 1.528x | 9/9 | 20.5% |
| 8 | 8 | 0.09990 | 1.055x | 0.966x | 8/9 | 21.5% |
| 8 | 16 | 0.09950 | 1.061x | 0.971x | 8/9 | 22.1% |
| 8 | 32 | 0.10243 | 1.060x | 0.985x | 8/9 | 25.0% |
| 16 | 0 | 0.05459 | 3.175x | 2.856x | 9/9 | 17.8% |
| 16 | 8 | 0.10066 | 1.976x | 1.810x | 9/9 | 18.5% |
| 16 | 16 | 0.10170 | 1.997x | 1.783x | 9/9 | 19.2% |
| 16 | 32 | 0.12005 | 1.619x | 1.544x | 9/9 | 22.1% |

An independent B8 C32 confirmation used 15 paired trials with 50 calls per
timing interval. It won 15/15 with `1.056x` median and `1.029x` minimum. This
confirms a real narrow B8 edge, but the earlier `0.985x` trial keeps the cell
experimental rather than promoted. B16 has a clear systems margin.

B4 was not rerun after this boundary was established. The refined selector
alone costs `0.097-0.102 ms`, already above the prior B4 FlashInfer exact
baseline near `0.075 ms`; adding route compilation and attention cannot make
that complete path positive. B4 therefore remains on proxy-only selection or
exact fallback until the refinement stages are fused.

Candidate width has little effect on selector latency from C8 through C32 at
B8. The dominant added cost is the extra stage/launch structure, not the
number of candidate tokens alone. A future refinement kernel should therefore
fuse exact candidate scoring with final selection or with the selected
attention producer instead of merely reducing C.

All eight phase cells matched the independently lowered selected-token
reference with maximum absolute error no larger than `0.00390625`. Synthetic
Gaussian recall is intentionally reported only as a kernel diagnostic. Real
Qwen captures previously showed the P4/refine-32 proxy close to block-max on
coverage, while even the stronger full block-max oracle still failed the
fragile-bucket model safety gate. No new model replay was run because this
systems implementation cannot overcome that known policy upper bound.

## Safety Boundary

The selected WGMMA kernel is exact over the chosen atoms. The selector is not
exact full-context attention. Existing Qwen stress analysis found that an
offline P4 extreme-support proxy with exact refinement over 32 candidates came
close to the block-QK oracle on coverage, but late-layer value/composition risk
still prevented adversarial promotion.

The two-stage systems path is now implemented and measured. It is useful at
B16 and sits on a narrow B8 boundary, but selector quality remains a policy
constraint. Unknown or failed request tiers remain on exact native attention.
The next meaningful selector optimization is stage fusion; the next meaningful
policy intervention is stronger than block-max selection, such as exact late
layers, a live verifier, or trained model adaptation.

## Artifacts

- `artifacts/h100/paged_query_selected_phase_20260826.json`
- `artifacts/h100/paged_query_selected_p8_20260826.json`
- `artifacts/h100/paged_query_selected_smoke_20260826.json`
- `artifacts/h100/paged_query_refined_phase_20260826.json`
- `artifacts/h100/paged_query_refined_b8_confirmation_20260826.json`
- `docs/qwen25_3b_dynamic_selector_findings.md`
