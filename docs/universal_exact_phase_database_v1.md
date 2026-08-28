# Universal Exact Phase Database v1

The exact phase database is the evidence boundary between GPU profilers and
runtime dispatch. Profilers may use different libraries and kernel launch
mechanisms, but they must produce the same immutable `BackendEvidence` schema.
The database compiler performs no timing itself.

## Why This Layer Exists

A benchmark name is not a resolved backend. PyTorch SDPA can select different
implementations, FlashInfer may choose FA2 or FA3, and a requested backend can
be unsupported for a particular mask, layout, or head dimension. A universal
comparison is valid only when each eligible baseline has an explicit outcome:

```text
measured     resolved backend, timing, correctness, workspace, confidence
unsupported  backend was attempted and rejected the semantic cell
error        backend was attempted but execution failed
invalid      measurement was rejected by the harness
```

Missing evidence is different from `unsupported`. If any baseline named by the
manifest has no outcome, the cell becomes `incomplete_baselines` and cannot
claim a fastest correct baseline.

## Evidence Contract

Each measured record includes:

```text
cell_id
provider: streamattn | external
requested_backend
resolved_backend
native
family_id and kernel_key for StreamAttn
environment fingerprint
workspace_bytes
supported_range
cold, p10, p50, p90, variance
process_count and sample_count
timed_allocation_count
confidence
correctness reference and errors
```

The environment fingerprint covers GPU architecture/name/UUID, driver, CUDA,
PyTorch, relevant libraries, and compilers. One architecture database cannot
mix fingerprints. Separate machines or software stacks therefore produce
separate reproducible artifacts rather than an ambiguous aggregate.

## Resolution Rules

For each manifest cell the compiler:

1. verifies that every eligible external baseline has an outcome;
2. ignores measurements that failed correctness or allocated in the timed loop;
3. selects the lowest-p50 valid external measurement, preserving both requested
   and resolved backend names;
4. selects the lowest-p50 valid native StreamAttn candidate only when it is no
   slower than the external baseline; otherwise the default route is an
   explicit external fallback;
5. preserves an explicit native selection when evaluating routing regret;
6. computes speedup against the fastest correct baseline;
7. computes routing regret against the fastest valid native-or-external route;
8. retains every input record, including failed candidates and negative cells.

If a registered native family exists but no valid native measurement is
available, the status is `native_unmeasured`. A measured native loss becomes an
`external_fallback`, while its losing native evidence remains in the database.

## Outputs

Run:

```bash
python benchmarks/compile_universal_exact_phase_db.py \
  path/to/evidence.json \
  --architectures sm90 sm100 \
  --output-dir phase_db \
  --source-commit <git-sha>
```

The compiler writes:

```text
phase_db/index.json
phase_db/sm80.json
phase_db/sm90.json
phase_db/sm100.json
```

The index records each database SHA-256. Architecture files contain the full
problem descriptors, selected plans, fastest baselines, speedups, routing
regret, confidence, workspace, supported ranges, acceptance summary, and all
raw evidence.

## Compiler v1 Acceptance

The database reports:

```text
semantic_coverage
telemetry_coverage
resolved_coverage
native_coverage
p90_routing_regret
zero_timed_loop_allocations
negative_cells
compiler_v1_pass
```

`compiler_v1_pass` requires complete semantic and baseline telemetry coverage,
a resolved exact path for every declared cell, zero timed-loop allocations on
selected measurements, and p90 routing regret within the manifest threshold.
It does not require StreamAttn to beat every baseline. Negative cells remain in
the artifact because they define the next kernel and routing work.

## Current Boundary

The first partial real-GPU calibration now exists. Four H100 cells compile to
three native routes and one external fallback; four B200 cells compile to
external fallbacks against stronger graph-captured baselines. The remaining
SM80/SM90/SM100 cells are explicitly unresolved. See [the calibration
report](universal_exact_calibration_20260828.md). StreamAttn therefore has a
working universal compiler and partial measured phase databases, not a
universal overall performance result.
