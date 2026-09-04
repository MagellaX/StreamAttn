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
- versioned macro and physical schedule IR.

Not yet implemented:

- real serving trace capture and boundary generators;
- critical-path resource DAG and architecture basis measurements;
- SM90 `M=2-64` exact micro-prefill kernels;
- mixed-ragged macro-plan timing and dispatch;
- a no-external-fallback H100 phase database.

The next implementation stage is the architecture basis-kernel/counter harness,
followed by the SM90 micro-prefill family. GPU results should not be attributed
to v2 until those measured artifacts exist.
