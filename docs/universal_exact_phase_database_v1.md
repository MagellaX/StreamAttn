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
4. selects the lowest-p50 valid native StreamAttn candidate unless an explicit
   routing choice is being evaluated;
5. computes speedup against the fastest correct baseline;
6. computes routing regret against the fastest valid StreamAttn candidate;
7. retains every input record, including failed candidates and negative cells.

If a registered native family exists but no valid native measurement is
available, the status is `native_unmeasured`. External fallback is allowed only
when the family registry declares that no native family covers the cell, as is
currently true for deterministic dropout backward.

## Outputs

Run:

```bash
python benchmarks/compile_universal_exact_phase_db.py \
  path/to/evidence.json \
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

The schema, resolver, compiler, CLI, and round-trip tests are implemented. The
committed 30-cell manifest has not yet been calibrated across real A100, H100,
and B200 processes using this schema. Until those evidence artifacts exist,
StreamAttn has a universal compiler contract rather than a universal overall
performance result.
