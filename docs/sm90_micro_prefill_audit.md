# SM90 micro-prefill: boundary and component audit

## Why this experiment

The previous 72-cell canary selected the fastest native candidate independently
in each measured cell. Its 1.342x geometric mean used Flash SDPA as the only
baseline. That is an oracle portfolio result, not public-dispatch performance,
holdout routing regret, or a victory over the fastest eligible exact backend.

The next measurement must answer three questions together:

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
the existing 64-row producer. This repair has CPU contract tests; GPU replay
and memory-sanitizer confirmation remain pending.

## Measurement contract

`profile_sm90_micro_prefill_audit.py` runs 52 cases per provider:

- 10 common anchors: M2/8/16/32/64, N16K, G8, D64/128, B1, Hq16;
- 10 common boundaries: M3/9/17/33/63, N320/448/4160/704/1216,
  B2, Hq32, G4, D64/128, including explicit irregular splits;
- 32 provider-specific expansion cases: M4/16/32/64, N4K/32K, G4/8,
  D64/128; B1/Hq16 on one provider and B2/Hq32 on the other.

There are 84 distinct shapes and 20 cross-provider replay shapes. All are
calibration data; none are advertised as held-out serving traces.

Each case records:

- full FP32 QK/softmax/PV reference with TF32 disabled;
- native and incumbent output error (max abs <= .04 and relative L2 <= .02);
- reconstructed natural-family LSE (absolute error <= .02);
- combined versus separate producer/merge output agreement;
- graph replay after changing Q, K and V, plus live tensor-allocation delta;
- rotated/reversed timing order, seven trials and 40 replays per trial;
- graph timings for both native candidates, forced Flash SDPA, forced
  FlashInfer FA2 and FA3, and the natural producer and merge independently;
- source/protocol fingerprints, package versions and GPU identity.

Allocation delta measures PyTorch live tensor bytes; it is not a proof about
all driver allocations. Isolated producer and merge times may not sum to the
combined replay time because cache residency and launch interactions differ.
These are component experiments, not the complete Nsight-counter basis suite.

The fastest **correct, measured** baseline wins. An unavailable or incorrect
backend stays visible and makes the baseline set incomplete. The incumbent
uses prepared NHD ragged KV with preparation excluded from timing. StreamAttn
uses its existing contiguous HND cache. This deliberately favors a prepared
incumbent; it is not a same-layout, end-to-end cache-conversion comparison.
Other eligible libraries still need coverage before a best-all-backends claim.

FlashInfer's pinned [v0.6.17 implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.17/flashinfer/prefill.py)
provides the forced backend and graph-buffer APIs used by the audit.

## Execution status, September 5, 2026

Both provider clients were launched concurrently. No GPU benchmark result was
produced:

- Lightning Nebius H100 job `job_01m1q44xvcq0n44xbzb34xq49h` failed before
  execution with `[ERROR]: job reconciliation failed`.
- A retry on the advertised available single-H100 bare-metal mapping,
  `job_01m1q4ccr41ep43wjqd882e27e`, failed with the same message.
- A minimal `nvidia-smi` probe, `job_01m1q4dmjak94gv3tesv8rthke`, also failed
  before execution. All three jobs were deleted; the two benchmark job status
  checks reported zero cost. The probe cost was not independently retrieved.
- The Modal launcher and `modal app list` could not connect to its API.
  A direct HTTPS probe also timed out. The launcher reported no app ID.

These are execution failures, not kernel failures or performance measurements.
The raw bare-metal launch failure is retained separately from benchmark data.

Rerun the two commands concurrently once provider execution is available:

```bash
python -X utf8 -m modal run benchmarks/modal_sm90_micro_prefill_audit.py
python -X utf8 benchmarks/run_lightning_sm90_micro_prefill_audit.py \
  --cloud-account lightning-baremetal --machine lit-h100-80gb-1
```

Each requests one H100, has a 45-minute remote limit and uses existing provider
authentication. The Lightning runner deletes its job in `finally`. Existing
successful local evidence files cannot be overwritten accidentally.

## What comes after results

Do not register either candidate solely from the old Flash SDPA oracle result.
Inspect incumbent choice and component timings first. A producer-dominated
loss motivates the asynchronous wider-row candidate; substantial merge cost
motivates an output-state/scheduling change. Neither has yet been established
by this audit. Then extend layouts, masks and ragged batches and evaluate real
trace-weighted routing, keeping the v1 evidence database unchanged.
