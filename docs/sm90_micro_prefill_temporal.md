# Does temporal overlap help without widening the tile?

StreamAttn's H100 micro-prefill work is full exact attention. No valid KV
position is dropped. This experiment changes the execution schedule, not the
attention definition, and remains outside public dispatch.

## Why this experiment

The first wider R128 smoke run passed the sampled output, LSE, split-composition,
and mutable-input graph checks, but lost on complete-call latency. At
`B1/M64/N4096/Hq16/G8/D128/C16`, the medians were:

| Implementation | Complete call |
| --- | ---: |
| Forced Flash SDPA | 19.94 us |
| Original R64 | 31.22 us |
| R128 serial | 49.46 us |
| R128 overlap | 51.99 us |

[Raw first smoke](../artifacts/gate0/sm90_micro_prefill_128_smoke_modal_h100_20260905.json).
The larger regression already exists without overlap. Fixing the additional
overlap penalty alone cannot make R128 competitive with R64, let alone Flash.

The new candidate therefore starts from R64. It preserves 128 threads, K64
tiles, two K stages and one V stage, SS-QK/RS-PV instructions, BF16 probability
fragments, FP32 state, balanced splits, and the original merge source verbatim.

## The dependency that changes

For each block, online softmax tracks the maximum `m`, denominator `l`, and
unnormalized output `n`:

```text
m_next = max(m, max(scores_next))
alpha  = exp(m - m_next)
l_next = alpha * l + sum(exp(scores_next - m_next))
n_next = alpha * n + exp(scores_next - m_next) @ V_next
```

The new scores and denominator can be computed while the previous PV updates
`n`. Rescaling `n`, replacing PV's register probability operand, or overwriting
its shared V buffer cannot happen until that PV group retires.

The temporal candidate commits next-QK then current-PV. A partial wait retires
QK before next-softmax; a full wait retires PV before output rescaling and
probability reuse. Prologue and tail handle one-tile splits separately. There
is only one output accumulator. The `drained` candidate keeps the same code
but fully retires both groups at the partial-wait point.

This is the dependency pattern used in FA3's intra-warpgroup overlap branch,
adapted to StreamAttn's existing operand and state layout. It is not a new
softmax formula or evidence that the compiler preserved useful overlap.
[Pinned FA3 mainloop](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp).

## Three causal anchors

Let `J = B * Hkv * ceil(M*G/R)`, producer CTAs `P = J*C`, and mean KV tiles
per CTA `L = (N/64)/C`. Then `P*L = J*N/64`. Increasing split count exchanges
pipeline length for parallel work; it cannot improve both.

| B | M | N | C | R | Producer CTAs | KV tiles/CTA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 4096 | 16 | 64 | 256 | 4 |
| 1 | 64 | 16384 | 16 | 64 | 256 | 16 |
| 2 | 64 | 4096 | 16 | 64 | 512 | 4 |

All three use Hq16/G8/D128. The second changes pipeline depth while retaining
CTA count; the third changes independent CTA work while retaining depth.
An optional C8 probe is separate and only warranted by a depth-dependent result.
Balanced nondivisible intervals are checked, including N4096/C24.

## What is measured

- Independent FP32 output and LSE on every head.
- Poisoned partial buffers followed by separate producer/merge execution.
- Mutated Q/K/V after graph capture, with zero live PyTorch allocation delta
  for native replay. This does not measure every CUDA driver allocation.
- Seven rotated/reversed paired trials, 40 graph replays each, fixed buffers
  and warmup. Complete, producer-only, and merge-only timings remain separate.
- Exact loaded binary and kernel symbols, build inputs, compiler versions,
  retained PTX, SASS, and runtime occupancy/resource queries.

Timing is unprofiled warm graph replay. Static instruction counts cannot prove
dynamic overlap or cache residency. Nsight measurements, when collected, must
label cache and replay controls explicitly; they are not interchangeable with
the promotion timing regime. No DRAM/L2 traffic reduction is inferred merely
from logical byte counts.

## Baselines stay independent

The external audit runs Torch Flash, FlashInfer FA2, FlashInfer FA3, standalone
FA3, cuDNN, and xFormers CUTLASS in fresh processes, each with both original
native controls. This avoids competing FA3 operator registrations. Ratios are
paired within a worker; raw latency rankings across workers are not treated as
a simultaneous fastest-baseline race. Unsupported, failed, and unavailable
backends remain explicit, not silently replaced by another implementation.

FlashInfer FA3 receives logical NHD views with the original physical HND
strides. This is a zero-copy view, not repacking. Wrapper launches and required
output handling stay inside baseline timing. The standalone FA3 image is a
pinned forward-only BF16 D64/D128 build; it is not evidence for FP16 or masks.

## Reproduce

On an already provisioned H100 with CUDA and CUTLASS:

```bash
python benchmarks/profile_sm90_micro_prefill_temporal.py \
  --suite smoke --cutlass-root /path/to/cutlass \
  --build-dir /tmp/temporal --output smoke.json
python benchmarks/profile_sm90_micro_prefill_temporal.py \
  --suite anchors --cutlass-root /path/to/cutlass \
  --build-dir /tmp/temporal --output anchors.json
python benchmarks/profile_sm90_micro_prefill_128.py \
  --suite dependency --protocol all --binary-diagnostics \
  --cutlass-root /path/to/cutlass --build-dir /tmp/r128 --output r128.json
python benchmarks/profile_sm90_micro_prefill_isolated_audit.py \
  --provider local --cohort smoke --cutlass-root /path/to/cutlass \
  --build-dir /tmp/baselines --output-json baselines.json
```

## H100 outcome

The three primary anchors passed FP32 output/LSE, split-composition and
mutable-input graph checks. None passed the paired performance comparison:

| B | N | Original R64 | Temporal | Fully drained | Flash SDPA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 31.13 us | 36.82 us | 36.96 us | 19.51 us |
| 1 | 16384 | 52.41 us | 63.51 us | 63.25 us | 42.12 us |
| 2 | 4096 | 55.43 us | 68.80 us | 65.97 us | 27.39 us |

[Raw anchors](../artifacts/gate0/sm90_r64_temporal_anchors_modal_h100_20260905.json)
and [compiled diagnostics](../artifacts/gate0/sm90_r64_temporal_anchors_modal_h100_20260905.diagnostics.zip).
The independent smoke also passed, including one-tile and nondivisible split
intervals, with the same direction of regression. D128 temporal and drained
both used 158 registers/thread, no local memory, and three potential resident
CTAs/SM. The original D128 producer binary reported 162 registers and no local
memory. Excess register count or spills alone cannot explain this result.

The separate R128 diagnostic did isolate a compiler effect. Its original
overlap symbol has C7514 and 42 static full hardware wait sites; the two
intended partial waits do not survive. The iteration-end drain removes C7514
for that exact symbol and retains three full/two partial hardware wait sites.
All versions remain at 254 registers/thread and two potential CTAs/SM.

R128 complete-call medians were 51.804 us overlap, 49.386 us drained, and
49.334 us serial. Drained beat overlap in 7/7 pairs, recovering roughly 98%
of its regression relative to serial, but still lost substantially to R64.
[Dependency artifact](../artifacts/gate0/sm90_micro_prefill_128_dependency_v2_modal_h100_20260905.json)
and [exact binaries/SASS](../artifacts/gate0/sm90_micro_prefill_128_dependency_v2_modal_h100_20260905.diagnostics.zip).
Static scheduling plus paired timing supports a compiler-serialization
diagnosis; it is not direct measurement of concurrent execution or stall time.

**Decision:** retain the original R64 control. Neither widening nor this
temporal schedule is promoted. No C8 sweep follows: increasing pipeline depth
at fixed CTA count did not reveal a winning temporal regime. Before another
performance design, measure the exposed producer stages and memory hierarchy,
with explicit warm/cold profiling conditions.

### What the R64 binary suggests researching

The D128 temporal loop retains a partial hardware wait, but only one of its
34 static loop EX2 sites lies between that wait and the full drain. Most
probability exponentials are scheduled later. Source placement therefore did
not establish the intended softmax/PV overlap. The new loop also rematerializes
thread/swizzle addresses and starts V copies later, losing some copy hiding
available in the original schedule. These are binary observations, not sampled
stall-cycle attribution.

Additional reciprocal sites are concentrated in setup and normalization, not
the hot KV loop. Both sources divide per output column; the temporal version
did not introduce a different epilogue formula. A reciprocal-only patch would
not address the observed repeated-loop work.

The next narrow *research question*, in service of the general compiler, is
whether preserving address temporaries and enforcing the intended independent
score work before full retirement can recover overlap without losing V-copy
hiding. First compare source-minimal drained lowering and inspect its generated
schedule. Use dynamic issue/memory counters to test the explanation, rather
than attributing latency directly to counts of static instructions. Persistent
P competing with address temporaries is a hypothesis, not a measured register
allocation cause. Semantic expansion need not wait for that research.

## Remaining engine work

This schedule experiment is not the complete engine. FP16, logical causal
append positions, masks, paging, genuinely ragged batches, and serving-trace
holdouts remain separate required work. The compiler should learn measured
relationships among state size, tile geometry, pipeline depth, parallel work,
and merge bytes, not a blanket rule that M64 requires a wider tile.
