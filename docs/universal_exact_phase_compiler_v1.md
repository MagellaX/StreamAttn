# Universal Exact Attention Phase Compiler v1

## Purpose

StreamAttn has native kernels that win in parts of the H100 exact decode and
B200 exact prefill surfaces. Those wins prove the physical designs, but manually
adding promotion dictionaries cannot establish overall engine superiority.

Phase Compiler v1 turns the repository's existing semantic planner and native
assets into the inputs of a reproducible offline compiler:

```text
valid workload manifest
  -> matching kernel families
  -> schedule candidates
  -> static resource legality
  -> compile and ingest resource reports
  -> correctness gate
  -> paired benchmark
  -> phase database
  -> allocation-free runtime plan
```

The repository implements the first four contracts through static legality and
the strict evidence-to-phase-database compiler. The first H100/B200 cells have
now been calibrated; the full manifest is still incomplete.

## Guarantee Invariant

The engine now names three distinct guarantees:

| Value | Meaning |
|---|---|
| `exact` | Full-context exact attention over all semantically valid KV tokens |
| `schedule_exact` | Exact arithmetic over an explicitly selected KV schedule |
| `distribution_verified` | Approximation accepted by model-output validation |

An `exact` problem cannot use a selected tile plan. A selected schedule must be
declared `schedule_exact` or `distribution_verified`. This prevents a dynamic
or reduced-work route from appearing as a universal exact kernel.

## Frozen Workload Surface

The committed manifest is
`benchmarks/manifests/universal_exact_v1.yaml`. Its 30 explicit cells cover:

| Surface | Cells | Purpose |
|---|---:|---|
| Real | 12 | Representative serving and training work |
| Boundary | 10 | Partial tiles, ragged rows, resource and regime transitions |
| Feature | 8 | Masks, windows, ALiBi, dropout, deterministic backward |

The aggregate includes SM80, SM90, and SM100; decode, prefill, and training;
FP16/BF16; D64/D128/D256; MHA/GQA; and contiguous/paged NHD/HND layouts.
Invalid products such as paged training and decode dropout are intentionally
absent.

Every cell records:

```text
semantic shape and features
trace / stratified / boundary weights
eligible exact baselines
dtype-specific correctness tolerance
```

The manifest must be changed before a benchmark campaign, not after results are
known. Losing cells remain part of the resulting artifact.

## Schedule IR

`ScheduleCandidate` contains instruction-shaping decisions only:

```text
algebra orientation
CTA ownership
query-head and query-position packing
KV tile and head-dimension staging
Q/KV splits and producer CTAs
load and MMA engines
register/shared/TMEM accumulation
warp groups and pipeline stages
scheduler and cluster shape
softmax, merge, epilogue, backward role
```

Runtime `B`, `M`, `N`, ragged lengths, and page tables remain in
`AttentionProblem`. Consequently a stable kernel key can serve many workload
cells instead of producing one binary per benchmark point.

## Resource Legality

The first architecture models cover SM80, SM90, and SM100. Before a candidate
is compiled or benchmarked they reject:

```text
thread-block overflow
shared-memory overflow
register-file overflow
illegal cluster shape
TMEM use on non-SM100 hardware
SM100 TMEM-column overflow
compiled register spills
zero estimated occupancy
```

The built-in limits are conservative pruning inputs. `CompiledKernelRecord`
exists so actual compiler version, binary digest, registers, shared memory,
TMEM, spills, and correctness status replace analytical estimates after build.

## Registered Families

The compiler starts from current assets:

```text
SM80 paged exact decode
SM90 transposed exact GQA decode
SM100 TGV paged exact decode
SM100 TGV exact causal GQA prefill
generic grouped-GQA Triton prefill
generic native Triton online-softmax forward
generic native no-dropout Triton training
explicit external exact fallback
```

The external fallback is registered but marked `native=false`. Therefore an
inspection can distinguish semantic coverage from StreamAttn-owned native
coverage and from optimized promotion. The frozen surface currently reports
one explicit native gap: dropout training backward uses the external exact
fallback. That gap is retained rather than being mislabeled as native support.

Run:

```bash
python benchmarks/inspect_universal_exact_manifest.py
```

The inspector validates the manifest and prints every cell's matching native
and fallback families.

## Acceptance Boundary

Compiler v1 is complete only when the phase database reaches:

```text
100% semantic coverage of this manifest
100% explicit backend telemetry
no silent fallback
p90 routing regret <= 5%
zero timed-loop allocation for planned paths
all negative cells retained
```

The later engine-level performance bar remains separate:

```text
trace-weighted geometric mean > 1.20x
stratified median > 1.15x
p10 >= 1.00x
exact correctness
no external fallback in the core matrix
```

The baseline resolver and phase-database schema are implemented in
`stream_attention/phase_database.py`. The first calibration covers four SM90
exact-decode and four SM100 exact-prefill cells. It retains one H100 native loss
and four B200 native losses against graph-captured baselines, routing each loss
externally. See [the calibration report](universal_exact_calibration_20260828.md).
The next stage fills SM80 and the remaining phase/feature cells while optimizing
the measured negative boundaries.
