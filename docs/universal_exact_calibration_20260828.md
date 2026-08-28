# Universal Exact Calibration: H100 and B200

This is the first real-GPU calibration of the frozen
`universal_exact_v1_20260828` manifest into the strict phase-database schema.
It covers four SM90 exact paged-decode cells and four SM100 exact causal-GQA
prefill cells. It is a partial database, not the 30-cell acceptance result.

## Evidence Rules

Every measured row records the GPU/software fingerprint, requested and
resolved backend, p10/p50/p90 timing, variance, correctness, workspace, and
timed allocations. Every eligible baseline has an explicit measured or
unsupported outcome.

Paged decode does not time a contiguous materialization for cuDNN or PyTorch
SDPA. Those backends are marked unsupported for the direct page-table
contract. FlashInfer is measured with a preplanned paged wrapper and
preallocated output.

For contiguous B200 prefill, eager SDPA-style APIs allocate their returned
output. Those measurements remain in the artifact with
`timed_allocation_count=1`. The harness also captures the same fixed-buffer
operation into a CUDA graph; successful replay is allocation-free and is
eligible as a baseline. Capture is excluded from timing.

## SM90 Results

The resolved external baseline was FlashInfer FA2 0.6.17. Timings below are
p50; speedup is baseline divided by StreamAttn.

| Manifest cell | StreamAttn | FlashInfer | Speedup | Compiled route |
|---|---:|---:|---:|---|
| B1, 32K, G8, D128, HND | `0.02282 ms` | `0.04134 ms` | `1.849x` | native |
| B4 ragged, 32K cap, G8, D128, HND | `0.05498 ms` | `0.07178 ms` | `1.306x` | native |
| B8, 64K, G4, D128, NHD | `0.70096 ms` | `0.73546 ms` | `1.049x` | native |
| B8 ragged, 64K cap, G4, D128, NHD | `0.48496 ms` | `0.47046 ms` | `0.970x` | external fallback |

The fourth cell is the useful boundary. Direct NHD G4 is legal and correct,
but the ragged 64K schedule does not recover its merge and fragmented-load
cost. The compiler retains the native loss and selects FlashInfer.

An initial run accidentally resolved StreamAttn to its generic Triton paged
backend because the H100 CUTLASS root was not exported into the remote
environment. That run is retained as
`sm90_calibration_lowering_failure.json`; it is not used by the phase database.

## SM100 Results

All three TGV tile variants passed an independent FP32 dense causal-GQA
reference. `h8_q2` was fastest in every tested cell. The table compares it
with the fastest correct allocation-free graph-replay baseline.

| Manifest cell | StreamAttn | Fastest baseline | Baseline | Speedup | Compiled route |
|---|---:|---:|---|---:|---|
| B1, S256, G8, D128 | `0.01152 ms` | `0.01040 ms` | cuDNN SDPA graph | `0.902x` | external fallback |
| B1, S384, G8, D128 | `0.01658 ms` | `0.01441 ms` | FlashInfer prefill graph | `0.869x` | external fallback |
| B1, S512, G8, D128 | `0.02274 ms` | `0.01446 ms` | cuDNN SDPA graph | `0.636x` | external fallback |
| B2, S128, G8, D128 | `0.01045 ms` | `0.01031 ms` | cuDNN SDPA graph | `0.987x` | external fallback |

This does not contradict the older eager-only result. Against eager APIs the
native fixed-output plan still removes substantial framework/allocation
overhead. It does show that the earlier `1.13x-1.73x` claims do not establish
an architecture-wide kernel win once the baseline receives an equivalent
fixed-address replay path.

Two failed harness attempts are retained:

- CUTLASS `main` was source-incompatible with the pinned kernel headers; the
  working calibration uses commit `7107b05535f8977f5ecb9d01ee203205b1fd9bc4`.
- the first CUDA-graph capture inspected output before its first replay. The
  timing was usable but correctness was not; the corrected harness replays and
  synchronizes before checking output.

Neither failed artifact participates in route selection.

## Partial Database

The generated files are:

```text
phase_db/calibration_v1/index.json
phase_db/calibration_v1/sm90.json
phase_db/calibration_v1/sm100.json
```

SM90 resolves 4 of 11 manifest cells: three native and one external fallback.
SM100 resolves 4 of 12: all four external fallbacks. Missing cells remain
`incomplete_baselines` or `native_unmeasured`; they are not inferred from
nearby shapes. Therefore `compiler_v1_pass=false` is the correct result.

## Decision

This calibration changes the next kernel work:

1. preserve the three measured SM90 native wins;
2. repair or route around ragged NHD/G4 merge cost;
3. treat B200 graph replay as the minimum baseline, then reduce TGV launch and
   epilogue cost or capture a routable native graph plan;
4. calibrate the SM80 surface and remaining prefill/train/feature cells without
   changing the frozen manifest.

Universal performance remains unproven. What is now proven is that the engine
can compile real positive and negative measurements into fail-closed routes.
