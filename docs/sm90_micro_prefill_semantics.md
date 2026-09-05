# Exact Micro-Prefill: Dtypes and Logical Positions

This extends the retained transposed and natural R64 families, not the slower
R128 or temporal experiments. The goal is to make the existing exact state
machines usable by more serving requests before building the mixed-batch path.
There is no seed policy, subset approximation, or public-route promotion here.

## Contract

Both `MicroPrefillPlan` and `NaturalMicroPrefillPlan` accept:

- Q `[B,M,Hq,D]` and contiguous HND K/V `[B,Hkv,N,D]`;
- native FP16 or BF16 inputs/output, with FP32 accumulation and partial states;
- `M=2..64`, `G=4/8`, `D=64/128`, positive `N` divisible by 64;
- noncausal attention, or `causal=True` with explicit device `int64` positions;
- up to `min(N/64,512)` splits, matching the merge's static workspace capacity.

For causal execution the visibility rule is exactly
`key_positions[b,j] <= query_positions[b,i]`. Neither array has to start at zero
or be physically sorted. All physical K/V rows are otherwise live. Logical
positions describe the mask; they do not apply RoPE, append cache entries, or
turn this into a speculative-tree ancestor mask.

An append request can be planned as follows (all construction is outside replay):

```python
import torch

from stream_attention.backends.sm90.micro_prefill import NaturalMicroPrefillPlan

B, M = q.shape[:2]
N = k.shape[2]
kp = torch.arange(N, device=q.device, dtype=torch.int64).expand(B, -1).contiguous()
qp = torch.arange(N - M, N, device=q.device, dtype=torch.int64).expand(B, -1).contiguous()
plan = NaturalMicroPrefillPlan.build(
    q, k, v, causal=True, query_positions=qp, key_positions=kp,
)
out = plan.run()
```

Supply per-request origins when caches do not start at logical zero. Buffers
and position values may change in place between graph replays, with stream
ordering supplied by the caller. Shape, dtype, storage addresses and mask kind
remain fixed for a plan. Changing those requires replanning.

The historical BF16 noncausal extension is unchanged. FP16 and position-causal
paths compile only the retained producers, helpers, and merges into an isolated
extension. FP16 uses `cutlass::half_t` and FP16 WGMMA operands; no full-cache dtype
conversion or K/V repacking is inserted. Checked source anchors fail rather than
silently composing different kernels when the shared definitions change.

## Empty-State Math

Each split publishes its normalized output `O_s` and log2 partition `L_s`.
The merge uses weights `w_s = 2**(L_s - max(L))` and returns
`sum(w_s * O_s) / sum(w_s)`. A split with no visible keys must publish
`O_s=0, L_s=-inf`. If every split is empty, the defined result is zero output
and `-inf` LSE. Explicit guards prevent `-inf - -inf` from becoming NaN.

This matters even with a nonempty cache: a future-only split, a permuted physical
cache, or a query before the cache origin can have no visible keys. The natural
producer still uses balanced integer split intervals, preserving the earlier
out-of-bounds preload fix.

## Verification Protocol

`profile_sm90_micro_prefill_semantics.py` defines 84 cases spanning both dtypes,
both head dimensions, B1/B2, Hq16/Hq32, G4/G8, M2/3/8/9/17/32/64, and
N64/320/448/4096/16384. These are stratified semantic cases, not a Cartesian
performance grid. Every case runs both families, first eagerly, then in a
captured graph after mutating Q/K/V and positions in place. Workspaces are
poisoned before each correctness check.

The independent FP32 reference constructs the explicit visibility relation,
computes full QK and PV with TF32 disabled, and checks output plus reconstructed
LSE. Cases include append alignment, reversed physical positions, offsets above
32-bit range, completely empty rows, one-visible-key rows, and split tails.
Native graph latency is recorded for diagnostics only. An implicit top-left
causal baseline is not timed as though it performed the same workload.

CPU CI checks contracts and reference edge cases. The CUDA source workflow
compiles six additional dtype/mask/D specializations without requiring a funded
GPU runner. Hardware runs remain necessary to validate numerical behavior.

## H100 Results

The full Lightning run passed **84/84 cases**; an independent Modal H100 replay
passed **24/24** with another seed. Each case checks two families and two input
states: 336 plus 96 output/LSE comparisons against FP32, respectively.

| Run | Cases | Worst BF16 output abs. error | Worst FP16 output abs. error |
| --- | ---: | ---: | ---: |
| Lightning, seed 6107 | 84/84 | 0.007780 | 0.0009873 |
| Modal, seed 9613 | 24/24 | 0.007626 | 0.0009751 |

Output tolerances are `atol=rtol=0.02` for BF16 and `0.003` for FP16;
LSE absolute tolerance is `0.005`. Exact means full attention over the defined
visible keys with normal floating-point rounding, not bitwise FP32 equality.
Both runs used PyTorch 2.7.1+cu128 and NVIDIA H100 80GB HBM3. Source hashes,
per-case errors, graph timing samples and case metadata are in the raw files:

- [Full matrix](../artifacts/gate0/sm90_micro_semantics_lightning_h100_20260905.json)
- [Independent replay](../artifacts/gate0/sm90_micro_semantics_modal_h100_20260905.json)

Lightning completed the kernel run successfully, but its transport inserted
newlines every 16 KiB inside the final JSON. The local collector initially
rejected it. The artifact was recovered by joining exactly those chunks;
the original log is retained and the collector has regression tests. No GPU
result was recomputed, guessed, or silently dropped during recovery.

## Reproduce on H100

Use a CUDA development environment with PyTorch, ninja, and the CUTLASS headers
resolved by `resolve_cutlass_root`. The measured build used CUDA 12.8; the
CPU-only source-build workflow separately checks CUDA 12.4 compatibility.

```bash
python benchmarks/profile_sm90_micro_prefill_semantics.py \
  --suite full --provider local --seed 6107 \
  --cutlass-root /path/to/cutlass \
  --build-dir /tmp/streamattn-micro-semantics \
  --output-json artifacts/gate0/micro_semantics_local.json
```

Use `--suite smoke` for the 24-case replay. The output path must be new so a
rerun cannot overwrite earlier evidence. The profiler raises on a correctness
failure and retains the partial artifact. Compilation and plan construction
are outside the recorded graph timings; these timings do not measure request
setup cost or establish a speedup against an external backend.

## Boundaries and Next Integration

This is a contiguous-cache, fixed-shape inference plan. Direct paged/ragged KV,
sliding/additive/tree masks, mixed-batch scheduling, and fastest-exact-baseline
holdout performance are still separate work. Arbitrary position loads also have
a cost: this first implementation does not claim that masking is free or that
M64 has become competitive.

Next, lower the same logical-position rule through direct page-table addressing
and per-request lengths. For affine positions, the compiler can eventually
prove wholly visible/invisible tile ranges and specialize the boundary. For
arbitrary physical positions it must retain explicit predicates unless stronger
metadata proves that optimization valid.
