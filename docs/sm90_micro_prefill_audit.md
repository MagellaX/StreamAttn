# SM90 micro-prefill: boundary and component audit

## Why this experiment

The previous 72-cell canary selected the fastest native candidate independently
in each measured cell. Its 1.342x geometric mean used Flash SDPA as the only
baseline. That is an oracle portfolio result, not public-dispatch performance,
holdout routing regret, or a victory over the fastest eligible exact backend.

The measurement answers three questions together:

1. Does exactness survive non-power-of-two shapes and changing graph inputs?
2. Does the apparent performance boundary survive forced FA2/FA3 comparison?
3. Is the natural family's time spent in the producer, merge, or both?

This serves Universal Inference v2's H100 vertical slice. No seed selection,
approximate attention, or public dispatch promotion is introduced.

## Boundary repair

The natural producer previously assigned `ceil(T/S)` KV tiles to each of `S`
splits, then preloaded the first tile unconditionally. For example, `T=5,S=4`
gave starts `0,2,4,6`; the final CTA read past the five-tile cache. This is a
source-level defect that the earlier power-of-two matrix did not exercise.

The new split interval is:

```text
begin(s) = floor(s * T / S)
end(s)   = floor((s + 1) * T / S)
```

For `1 <= S <= T`, every interval is nonempty, adjacent intervals meet exactly,
and their lengths differ by at most one tile. Their union is the full cache,
so the same normalized partial-output/LSE merge computes full-context attention
with the usual floating-point rounding differences. Query padding still uses
the existing 64-row producer. CPU contract tests and the irregular GPU smoke
now pass. A dedicated memory-sanitizer run remains pending.

## Measurement contract

`profile_sm90_micro_prefill_audit.py` can generate 52 cases per provider:

- 10 common anchors: M2/8/16/32/64, N16K, G8, D64/128, B1, Hq16;
- 10 common boundaries: M3/9/17/33/63, N320/448/4160/704/1216,
  B2, Hq32, G4, D64/128, including explicit irregular splits;
- 32 provider-specific expansion cases: M4/16/32/64, N4K/32K, G4/8,
  D64/128; B1/Hq16 on one provider and B2/Hq32 on the other.

There are 84 distinct shapes and 20 cross-provider replay shapes. All are
calibration data; none are advertised as held-out serving traces.
This larger matrix is not complete. The isolated audit below ran four smoke
cases per worker: B1/M2/N16K/G8/D128; B2/M3/N320/G4/D128; and B1/M64/N4K/G8
with D64 and D128.

Each case records:

- full FP32 QK/softmax/PV reference with TF32 disabled;
- native and incumbent output error (max abs <= .04 and relative L2 <= .02);
- reconstructed natural-family LSE (absolute error <= .02);
- combined versus separate producer/merge output agreement;
- graph replay after changing Q, K and V, plus live tensor-allocation delta;
- rotated/reversed timing order, seven trials and 40 replays per trial;
- graph timings for both native candidates and a selected external backend,
  plus the natural producer and merge independently;
- source/protocol fingerprints, package and loaded-binary hashes, and GPU
  identity.

Allocation delta measures PyTorch live tensor bytes; it is not a proof about
all driver allocations. Isolated producer and merge times may not sum to the
combined replay time because cache residency and launch interactions differ.
These are component experiments, not the complete Nsight-counter basis suite.

The resolver accepts only **correct, measured, revision-matched** baselines.
Unavailable and incorrect backends stay visible. The isolated supervisor runs
Torch Flash, FlashInfer FA2/FA3, standalone FA3, cuDNN, and xFormers CUTLASS in
separate processes, each with the native controls. It does not choose a global
winner by comparing absolute latencies from different workers.

Every adapter reads the original HND storage. FlashInfer FA3 receives a
logical NHD transpose view with unchanged physical strides, not a repack.
Preparation is excluded for both planned native and baseline calls; wrapper
launches, per-request loops, and required output handling remain timed.
The pinned [FlashInfer v0.6.13 implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.13/flashinfer/prefill.py)
provides the forced backend and graph-buffer APIs. Standalone FA3 is built
from v2.8.3, and xFormers is 0.0.31. These choices are recorded per worker.

## Execution status, September 5, 2026

The isolated Modal H100 audit completed all six workers without the previous
FA3 namespace collision. [Raw audit](../artifacts/gate0/sm90_micro_prefill_isolated_audit_modal_h100_20260905.json).
Its first pass exposed two harness defects rather than competitor failures:
synthetic Torch module filenames blocked provenance resolution, and FA2's
M64/D128 plan needed more than the original 128 MiB workspace. The former is
fixed with a fresh three-worker replay; the latter now has a 256 MiB planned
workspace and a dedicated replay. Original artifacts remain unchanged.

The [FA2 workspace replay](../artifacts/gate0/sm90_micro_prefill_isolated_fa2_workspace_replay_modal_h100_20260905.json)
passed and resolved all four cells. Its paired baseline/native ratios were
0.976x (M2), 2.149x (the M3 boundary), 0.756x (M64/D64), and 0.572x
(M64/D128). A ratio below one means the external baseline was faster.
This confirms the M64 performance gap after removing the harness allocation
limit; it does not justify promoting from the earlier Flash-only matrix.

[Provenance replay](../artifacts/gate0/sm90_micro_prefill_isolated_provenance_replay_modal_h100_20260905.json)
resolved all four cells for Torch Flash, cuDNN, and xFormers CUTLASS. The
xFormers record includes the loaded ATen CUDA implementation, not just its
Python wrapper. No adapter switches to another backend after failure.

Standalone FA3 passed all four cells. Native transposed beat it at M2 and the
M3 boundary by about 1.09x in paired trials, but native natural lost at both
M64 cells. At M64/D128, FA3 took 15.374 us and natural R64 took 31.097 us.
This is a stronger measured target than Flash SDPA alone. Very large ratios
against forced xFormers or a particular FlashInfer wrapper path are not
reported as victories over the fastest attention engine.

Lightning job `job_01m1rc2cbfjj4sk48p54fvazbq` again failed before execution
with `job reconciliation failed`, reported zero cost, and was deleted. Thus,
these are H100 results from Modal, not successful cross-provider validation.
[Launch-failure artifact](../artifacts/gate0/sm90_micro_prefill_isolated_lightning_h100_20260905.failure.json).

Run the collision-free audit on an already provisioned H100:

```bash
python benchmarks/profile_sm90_micro_prefill_isolated_audit.py \
  --provider local --cohort smoke --cutlass-root /path/to/cutlass \
  --build-dir /tmp/audit --output-json audit.json
python benchmarks/summarize_sm90_micro_prefill_audit.py audit.json
```

Each external worker has a bounded timeout and retains partial results and
logs. Existing evidence files cannot be overwritten accidentally.

## What comes after results

Do not register either candidate solely from the old Flash SDPA oracle result.
Both [widening and R64 temporal overlap](sm90_micro_prefill_temporal.md) have now
been tested and rejected as performance promotions. Preserve the original
controls. Profile actual exposed producer work before another physical-family
redesign, while extending layouts, masks and ragged batches toward real
trace-weighted routing. The v1 evidence database remains unchanged.
