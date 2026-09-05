# Testing a wider exact micro-prefill family

This experiment continues Universal Inference v2's H100 serving-batch plan.
It does not select or skip tokens: every valid KV position participates in
QK, online softmax, and PV. The open question is how to execute that work more
efficiently when the 64-row family loses at M64 and short KV lengths.

## What 128 rows means

A row packs one query position and one GQA head. With G8, 128 rows cover
16 query positions; with G4 they cover 32. A single 128-thread warpgroup
processes two 64-row fragments and shares their KV loads. This is not a
128-query-token tile and not a dedicated-producer warpgroup design.

The candidate has three protocols using the same data layout and state ABI:

- `serial` retires each QK/PV group before proceeding;
- `overlap` tries to process one fragment's softmax while the other fragment's
  matrix operation is outstanding, with explicit group-retirement fences.
- `overlap_drained` adds a full retirement at the iteration boundary while
  retaining the partial wait inside the iteration. It diagnoses loop-carried
  dependencies rather than replacing every partial wait with a full drain.

Both retain two K stages, one V stage, FP32 accumulators, and register-source
BF16 probabilities for PV. Shared storage is `640 * D` bytes: 40 KiB for D64
and 80 KiB for D128. The second fragment adds register pressure, so reducing
memory traffic alone does not establish a speedup.

## Exact state and controlled comparisons

Each split owns the balanced interval
`[floor(s*T/S), floor((s+1)*T/S))` of 64-token KV tiles. Partial outputs are
normalized FP32 values with base-2 LSE. The merge weights them by
`exp2(lse_split - max_lse)` and returns a natural-log LSE. For S1, an optional
direct epilogue avoids partial-state allocation and the merge launch entirely.

The default comparison matches the old 64-row family's split count. Retaining
its 256-CTA target after doubling rows would double splits, confounding reuse
with parallelism and extra merge work. The canary therefore exposes matched
S1/8/16/32, serial versus overlap, combined versus isolated stages, and S1
partial versus direct output as distinct measurements.

The profiler checks output and LSE against independent FP32 full attention,
reconstructs LSE from partials, poisons intermediate buffers before composition
checks, and mutates Q/K/V before graph replay. Live tensor allocation deltas
are recorded; these are not a census of all CUDA driver allocations.

## First H100 resource evidence

The initial smoke resource run compiled both D64 and D128 on H100:

| Producer | Registers/thread | Shared memory | Local memory/thread | Resident CTAs/SM |
| --- | ---: | ---: | ---: | ---: |
| D64 G4 overlap, partial | 166 | 40 KiB | 0 | 3 |
| D64 G4 serial, partial | 168 | 40 KiB | 0 | 3 |
| D128 G8 overlap/serial, partial | 254 | 80 KiB | 0 | 2 |
| D128 G4 overlap, direct | 240 | 80 KiB | 0 | 2 |
| D128 G4 serial, direct | 247 | 80 KiB | 0 | 2 |

The merge reported 33 registers, 2,176 shared bytes, no local memory, and
12 resident CTAs/SM. [Raw resource artifact](../artifacts/gate0/sm90_micro_prefill_128_resources_modal_h100_20260905.json).

These are compiled resource measurements; the subsequent
[smoke run](../artifacts/gate0/sm90_micro_prefill_128_smoke_modal_h100_20260905.json)
passed all three sampled numerical/replay cases but lost to both retained R64
and Flash SDPA. At B1/M64/N4K/G8/D128/C16, serial took 49.46 us and overlap
51.99 us, versus R64 31.22 us and Flash 19.94 us. This family is not promoted.
The CUDA 12.8 compiler also emitted `C7514` serialization diagnostics for the
overlap variants and inserted `C7519` fences. Consequently, the source-level
schedule alone is not proof that useful MMA/softmax overlap survives lowering.
The added dependency diagnostic retains generated sources, PTX, binaries,
exact-symbol SASS, resources, and compiler versions. Warning disappearance
alone cannot settle the performance claim.

## Reproduction and boundary

On an already provisioned H100:

```bash
python benchmarks/profile_sm90_micro_prefill_128.py --mode resources --suite smoke
python benchmarks/profile_sm90_micro_prefill_128.py --suite smoke --output smoke.json
python benchmarks/profile_sm90_micro_prefill_128.py --suite canary --matches-splits --output canary.json
python benchmarks/profile_sm90_micro_prefill_128.py --suite boundary --output boundary.json
```

The candidate remains isolated from public dispatch. Its current semantic
contract is BF16, contiguous HND KV, noncausal G4/G8, D64/D128, and M2-64.
FP16, masks, paging and genuinely ragged batches are subsequent lowerings of
the same exact workload contract, not claims established by these canaries.
