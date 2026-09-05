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
- measured basis-operation adapter kernels and counter artifacts;
- critical-path resource DAG calibrated from those measurements;
- fastest-exact-baseline resolution for the micro-prefill matrix;
- FP16, paged/ragged, causal/sliding, and additive-mask micro-prefill lowering;
- a competitive `M=64`/short-K physical family;
- mixed-ragged macro-plan timing and dispatch;
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

Expand the same semantic family to paged/ragged storage and masks independently
of the producer research, then measure mixed batches and holdout routing regret.
The goal remains the complete H100 vertical slice, not another per-shape
whitelist or an approximate seed route.
