# Universal Inference v2

## Purpose

Universal Inference v2 changes the compiler's unit of work from one rectangular
attention cell to one complete serving batch. It is the foundation for a single
exact engine that can choose different execution plans for decode, speculative
verification, micro-prefill, full prefill, and mixed ragged traffic.

The first proof target is an H100 vertical slice:

```text
M=1       scalar decode
M=2-8     speculative verification
M=9-64    micro-prefill
M>=65     prefill
mixed     heterogeneous ragged serving batches
```

This contract does not replace Universal Exact v1. The v1 manifest, kernel
keys, evidence, and phase databases remain frozen historical calibration and
regression records.

## Batch Contract

`AttentionBatchV2` describes the whole batch, including per-request query and
KV lengths, request phase, shared-prefix identity, exact page tables, final-page
length, and speculative-tree shape. Attention geometry, numerical formats,
storage, masking, cache updates, execution mode, workspace, and optimization
objective are batch-level semantics.

The validator rejects ambiguous representations. Examples include a paged KV
extent that disagrees with its page table, a CUDA-graph batch larger than its
captured capacity, an MHA/GQA/MQA label inconsistent with `Hq/Hkv`, or a
homogeneous phase label over heterogeneous requests.

The workload manifest is
[`benchmarks/manifests/universal_inference_v2.yaml`](../benchmarks/manifests/universal_inference_v2.yaml).
It composes three evidence sources instead of enumerating a giant Cartesian
matrix:

```text
serving traces       distribution weight
stratified coverage  balanced semantic coverage
boundary generators  tile, page, resource, and scheduler transitions
```

## Frozen Trace Evidence

`InferenceTraceRecord` wraps a validated batch with source identity, a canonical
SHA-256 workload fingerprint, and a frozen calibration or holdout assignment.
The assignment is a stable hash of the record identity and manifest salt. It is
independent of row order and corpus growth.

Re-importing an unchanged record is idempotent. Reusing its identity with a
different workload or source fails. This prevents tuning data from silently
moving into the final route-regret set.

```bash
python benchmarks/import_universal_inference_trace.py raw.jsonl \
  --output benchmarks/traces/universal_inference_v2.jsonl \
  --summary-json artifacts/universal_inference_v2_trace_summary.json
```

## Exact Baseline Resolution

The baseline resolver applies two separate gates:

1. Resolve direct semantic compatibility for the exact workload.
2. Choose the fastest correctness-passed measurement among eligible backends.

An implementation is rejected with explicit reasons such as `cache_kind`,
`mixed_batch`, `speculative_tree`, or `requires_layout_conversion`. A timing can
participate only when its workload fingerprint and backend revision match and,
for graph workloads, it measured graph replay.

The initial declarative capability registry is
[`benchmarks/manifests/exact_baselines_v2.yaml`](../benchmarks/manifests/exact_baselines_v2.yaml).
It is not performance evidence. Every winning baseline still requires an
immutable environment and timing artifact.

## Hierarchical Schedule IR

The v2 planner has two independently keyed levels:

```text
macro plan
  unified persistent, split phase, query-length cohorts,
  static rectangular, or prefix-sum ragged

physical schedule
  QK/PV operand source, producer and barrier topology,
  consumer overlap, task granularity, tile/split geometry,
  load/MMA engine, accumulator space, merge, and epilogue
```

This allows the compiler to compare one mixed launch with two or more
specialized launches. It also captures the state-machine differences that tile
dimensions alone missed in the recent SM90 experiments. The v2 keys are
separate from `ScheduleCandidate.kernel_key`, preserving every v1 artifact.

## Current Status

Implemented and CPU-tested:

- whole-batch exact workload schema and validation;
- trace, stratified, and boundary source contract;
- stable calibration/holdout trace partition;
- immutable workload fingerprints;
- direct exact-baseline eligibility and measured winner resolution;
- versioned macro and physical schedule IR;
- a versioned SM90 architecture-basis suite with 6 serving anchors, 14
  operation floors, required Nsight Compute counters, immutable adapter output,
  and environment fingerprints;
- two exact SM90 `M=2-64` candidate families: transposed query/head groups and
  natural 64-row query/GQA packing with exact split-state merging.

Initial H100 canary evidence covers 72 noncausal, contiguous-HND BF16 cells over
`M={2,4,8,16,32,64}`, `N={4K,16K,32K}`, `G={4,8}`, and `D={64,128}`. Both
families passed the sampled numerical checks in every cell. Selecting the faster StreamAttn family per
cell produced a `1.342x` geometric mean against graph-captured Flash SDPA and
won the paired Flash gate in 53/72 cells. The family boundary is material:

```text
M=2,4       transposed family dominates; 24/24 paired wins
M=8         mixed family choice; 12/12 paired wins
M=16        natural family usually wins; 10/12 paired wins
M=32        natural family usually wins; 7/12 paired wins
M=64        natural family always selected; 0/12 paired wins
```

This is a canary result, not compiler promotion. Flash SDPA was the only timed
baseline, FP16/paged/ragged/masked variants were not included, and the `M=64`
boundary remains below parity. The result nevertheless validates the central
v2 design: one semantic workload requires multiple physical families, and the
compiler must learn their crossover rather than use a global query-length rule.

Not yet implemented:

- real serving trace capture and boundary generators;
- measured basis-operation adapter kernels beyond the retained-producer counter captures;
- critical-path resource DAG calibrated from those measurements;
- full-matrix fastest-exact-baseline measurements, beyond the isolated adapter audit;
- sliding and additive-mask micro-prefill lowering;
- a competitive `M=64`/short-K physical family;
- mixed-ragged macro-plan search, holdout timing and dispatch;
- a no-external-fallback H100 phase database.

The cross-provider [micro-prefill audit](sm90_micro_prefill_audit.md) now implements
FP32 reference checks, forced FA2/FA3 comparisons and isolated natural-family
producer/merge timing. It includes irregular lengths and larger batches/head
counts. A fresh-process worker for each external backend avoids standalone and
vendored FA3 namespace collisions. Loaded binaries and interfaces are bound to
each worker's resolver revision; paired ratios remain within that worker.

The initial 128-row H100 smoke passed sampled output/LSE, split composition,
and mutable-input graph checks but lost on latency. At B1/M64/N4K/G8/D128/C16,
R128 serial was 49.46 us, overlap 51.99 us, retained R64 31.22 us, and Flash
SDPA 19.94 us. D128 R128 used 254 registers/thread. A wider tile is therefore
not the default answer to M64. [Detailed R128 experiment](sm90_micro_prefill_128.md).

The [R64 temporal experiment](sm90_micro_prefill_temporal.md) retained one output
state with separate pipeline-length and concurrency anchors. It passed the
sampled numerical/replay checks but lost at all three anchors: temporal
36.82/63.51/68.80 us versus control 31.13/52.41/55.43 us. A separate R128
footer-drain diagnostic removed exact-symbol C7514 serialization and recovered
its smaller overlap regression, but did not beat serial R128 or original R64.
Neither experiment registers a new public route. Compiler-visible overlap is
not sufficient evidence of a profitable schedule.

The isolated four-case baseline audit now resolves all six external adapters
after provenance and workspace repairs. At M64/D128, standalone FA3 took
15.37 us versus 31.10 us for original R64 in its paired worker. This is a
stronger competitive target than Flash SDPA alone, not a complete shape-matrix
or holdout result.

The retained R64 and transposed families now support native FP16/BF16 and
explicit int64 causal Q/K positions. The full 84-case Lightning H100 matrix
and independent 24-case Modal H100 replay passed all output/LSE checks,
including in-place graph input changes and empty visibility. This removes the
BF16/noncausal-only functional boundary for contiguous micro-prefill; it does
not claim whole-matrix performance promotion. See
[semantics and hardware evidence](sm90_micro_prefill_semantics.md).

A source-minimal R64 denominator-reduction ablation measured paired speedups
of 1.013x, 1.100x and 0.987x at three concurrency/depth anchors. Its mixed result
does not justify a universal switch. The retained producer remains unchanged;
[kernel research](sm90_kernel_research_20260905.md) records the math, upstream
sources, and the completed six-launch counter follow-up. Resource allocation
was unchanged, instruction savings were 0.34-1.00%, and eligible-warp supply
remained low. Source-correlated follow-up identifies scalar Q staging as
the strongest short-K load-dependency signal in that contiguous kernel.
It motivates a bounded staging test, but does not establish the bottleneck
inside the later paged producer.
See [source attribution](sm90_micro_prefill_mixed.md#source-level-producer-evidence).

Direct page-16 micro-prefill now extends both retained families to HND/NHD,
FP16/BF16, independent query/KV lengths and mutable device page tables. The
corrected Lightning H100 matrix passed 144/144 cases and the independent Modal
replay passed 48/48, including poisoned tails, shared prefixes and empty rows.
The experiment is documented in [paged micro-prefill](sm90_micro_prefill_paged.md).
It adds no KV gather/repack and does not register a new public dispatcher route.

The [mixed-ragged comparison](sm90_micro_prefill_mixed.md) now times the retained
rectangular plan against compatible exact paged FA2/FA3, with separate padded
and packed query contracts. It exposes a substantial performance gap and
empty rectangular work; functional coverage is not macro-schedule promotion.
An opt-in [compact work-proportional assignment](sm90_compact_ragged_schedule.md)
now keeps fixed-length planned tasks separate from the mutable rectangular path.
The full 48-case run improves that control by about 2.15-2.18x but still loses
to FlashInfer overall. Native packed output scheduling and holdout routing
remain uncompleted work. Further masks and the bounded Q-staging ablation
remain independent of that integration.
The independent 24-case replay now reproduces the compact scheduling gain and
remaining loss. A [three-arm affine-causal ablation](sm90_affine_causal_ablation.md)
now tests index masking and a fully visible tile fast path. The motivating signal:
matched-shape explicit-position causal latency is 1.70x noncausal in
the full run. Preserve arbitrary-position support; do not assume affine masks.
Both 24-case H100 runs passed and reproduced a 1.86-1.87x padded improvement
over compact causal control, but only 0.70x versus FlashInfer. Interior masking
adds about 9% beyond index masking. Retain this experimental candidate;
public dispatch is unchanged.

The [post-affine attribution](sm90_post_affine_attribution.md) now separates
producer, merge and interface costs on 24 discovery and 24 independent holdout
cases, all passing output/LSE and component checks. Separate Nsight captures
cover 27 actual launches. KV-major task order gives no general gain; a two-tile
floor's roughly 1% discovery improvement disappears on holdout. Neither is
promoted and default schedules are unchanged.

Large D128 traces reproduce a producer-only cost roughly twice the complete
packed FlashInfer call. Merge and copy deletion cannot plausibly be the sole
repair under that measured cost structure; isolated timings are not additive.
The next primary experiment must attribute the current paged producer's Q/K/V
staging, address work and QK/softmax/PV dependencies against the actual winning
FA2 kernel, then test the smallest justified reduction. Low eligible-warps
counts alone do not distinguish the two producers and do not justify another
R128 or persistent-kernel rewrite. Native packed offsets and live-row merge
ownership remain secondary simplifications, not a claimed solution to the
large-trace gap.
The goal remains the complete H100 vertical slice, not another per-shape
whitelist or an approximate seed route.

The [current paged-source capture](sm90_paged_source_attribution.md) now supplies
useful/issued work, DRAM/L2 traffic, executed instructions and native CUDA/SASS
correlation. The larger traces have similar DRAM reads and tensor tile work to
FA2 but substantially more native instructions. Scalar Q staging is the largest
individual long-scoreboard location; paged addressing/copy issue is the larger
aggregate instruction region. Vector Q copies now pass 24 discovery and 24
independent holdout cases with that single-state producer and fixed schedule.
They reduce Q-region instructions 86.6% and total producer instructions 7.35%
without extra registers or shared storage. Padded/packed complete-call gains
over the native control are 1.100x/1.065x in discovery and 1.117x/1.081x on holdout.
Every cell median improves, but three warm comparisons have losing individual
pairs; no per-cell exceptions were fitted. Packed calls still lose to FA2, so
this is a retained experimental improvement, not public promotion.
The actual FA2 binary lacks CUDA
lineinfo, so its aggregate measurements must not be presented as source-level
stall attribution. Producer-attributed stalls are also unavailable in this
export.

The [D128 page-pair ablation](sm90_page_pair_reuse.md) now passes its compiled-work
and complete-call gates. It resolves one live page/address for two copies,
preserving their individual predicates, destinations and completion ordering.
No descriptor bank, new shared state, split change or approximate attention
was introduced. Executed producer instructions fall from 49.00M to 24.68M;
KV-copy and tensor-op counts and the register/shared allocation stay unchanged.

The complete Modal H100 regression/boundary run passes 40/40 cases and the
independent Lightning H100 holdout passes 24/24. D128's original regression
cases improve 1.509x padded / 1.445x packed over vector Q; the new holdout improves
1.554x / 1.503x. All 896 D128 paired comparisons across both interfaces and
warm/perturbed modes improve; unchanged D64 controls remain near 1.0x.
The report separately retains short boundary results and both initial harness
failures. No new selector or public route is fitted from these results.

This removes about two-thirds of the native-to-FA2 gap in the two representative
heterogeneous packed cases, not across the whole engine. Packed D128 holdout
still reaches only 0.750x versus FlashInfer. Its representative isolated producer
is 100.17 us versus FA2's 81.30 us complete call; isolated timings are not additive.
Next reconcile the remaining producer address/layout and arithmetic dependencies
with the actual FA2 binary before selecting another local reduction. Native
packed offsets remain useful integration work, but cannot be treated as the
entire producer repair. Keep the retained R64 state machine and fixed schedule.
