# Kernel Research: Dependencies, Masks, and Generalization

Research accompanies the FP16/causal-append implementation. Sources are useful
when they identify a mechanism and a falsifiable experiment, not because of
their language or obscurity. This pass searched Chinese and Japanese GPU
material, patent descriptions, upstream kernels, and compiler reports. No
patent or blog supplied a demonstrated drop-in speedup for StreamAttn.

## Strongest Lead: Compiler Loop Dependencies

NVIDIA's engineering reply for **NVBUG 5431330** confirms overly conservative
register-use diagnostics in WGMMA loops mixing partial waits and other GMMA
groups. The supplied CUDA 12.8 reproducer changes behavior with loop structure;
NVIDIA describes a full wait as a workaround. A user also reports the issue on
CUDA 13.0, so changing toolchains alone is not an established repair.
[Original report and engineering response](https://forums.developer.nvidia.com/t/ptxas-mysterious-warning-for-wgmma-mma-async-instruction-serialization/340610).

**Our inference:** this is consistent with the prior R128 footer-drain result,
but does not establish that every barrier in our binary has this cause. Nor
does it close the much larger R64-versus-FA3 gap.

**Useful next experiment:** preserve the original R64 load layout, arithmetic,
split count and epilogue; change only the lifetime/control-flow representation
around one outstanding PV group. Compare PTX, exact-symbol SASS, compiler-added
waits, and dynamic stalls. A bounded two-iteration loop plus a tail is a
candidate, not a request to unroll the entire context or specialize every N.
Reject it if register pressure, instruction footprint, or complete latency
regresses even when warning counts improve.

## Operand Lifetimes Are a Correctness Constraint

The PTX ISA requires completion of the relevant WGMMA group before reading or
overwriting its accumulator or register-A fragments. `wait_group 1` means at
most one recent group remains outstanding; it does not name a particular
matrix. Group order and ownership must make the intended dependency explicit.
All warpgroup threads must execute aligned synchronization consistently.
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group).

For the intended overlap, QK-next may finish while PV-current is outstanding.
Softmax-next may run on its completed score fragment, but the old P operand and
O accumulator cannot be reused until PV-current completes. Reduced waiting is
useful only when the compiled lifetime graph still exposes this independence.

## What FA3 Actually Does

The pinned v2.8.3 Hopper mainloop separates K and V pipeline state, submits
QK-next and PV-current, partially waits, performs next-score work, and only then
fully waits before replacing P. It also splits mask-boundary iterations from
unmasked interior iterations. Its comments explicitly discuss register-bandwidth
cost for register-resident Q.
[FA3 mainloop, v2.8.3](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp#L1056-L1164).

**Inference for our generalization:** generic masks need not impose identical
instruction cost on every tile. Affine append positions can justify boundary
specialization; arbitrary reordered positions cannot. Proving tile visibility
from metadata is exact semantic elimination, not approximate attention skipping.
Test both paths on identical visibility sets and include metadata preparation
and extra launches in the complete-call cost.

The upstream mask implements length-offset causal alignment, rather than
blindly comparing local query and key indices.
[FA3 mask](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/hopper/mask.h#L71-L95).
Our new contract instead takes explicit int64 Q/K positions so physical cache
order and logical visibility are independently testable.

The upstream softmax handles all-masked rows explicitly and delays cross-thread
sum reduction until final normalization.
[FA3 softmax](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/hopper/softmax.h#L55-L139).
Our natural R64 baseline reduces the sum each tile. This led to the bounded
ablation below; it does not by itself solve memory/MMA overlap.

## Tested Lead: Deferred Denominator Reduction

For lane-local denominator `l_a`, tile rescaling uses the shared row maximum:
`l_a' = alpha * l_a + sum_local(exp(score - new_max))`.
Because `alpha` is identical across the row's lanes,
`sum_a(alpha * l_a) = alpha * sum_a(l_a)`. The lane sums can therefore be combined
once at split finalization. Maxima still synchronize each tile, P still enters
the same PV MMA, and the split-state merge is unchanged. Floating-point
association changes, so output and LSE were checked rather than assumed equal.

The source test reverses the two edits and reproduces the control source
exactly. The experiment used BF16/D128/G8/M64, 16 splits, two isolated builds,
FP32 output/LSE checks before and after input mutations, and nine alternating
paired trials of 200 graph replays on one Modal H100.

| Batch / KV length | Median control / candidate | Winning pairs |
| --- | ---: | ---: |
| B1 / 4K | 1.013x | 9/9 |
| B1 / 16K | 1.100x | 9/9 |
| B2 / 4K | 0.987x | 0/9 |

All three anchors passed correctness. The deeper pipeline gains more, but
doubling concurrency reverses the sign. This is a useful resource/scheduling
signal, **not a general improvement or a FlashInfer/FA3 victory**. The candidate
remains in the benchmark only; retained production sources are unchanged.
[Raw paired evidence](../artifacts/gate0/sm90_micro_deferred_sum_modal_h100_20260905.json).

Nsight Compute was present at `/usr/local/cuda/bin/ncu`; dynamic counters were
not collected in this run. The next attribution should compare shuffle issue,
register occupancy, barrier stalls and complete producer/merge latency at all
three anchors. Fewer source reductions alone do not explain the B2 regression.

For that follow-up, collect launch resources, scheduler issue/eligible-warps,
instruction mix and memory traffic before interpreting sampled stall reasons.
NVIDIA cautions that stalls matter for throughput when schedulers fail to issue;
counter replay can also change cache state and serialize launches. Keep the
uninstrumented paired graph measurements as the latency result, and use a
separate profile to explain it. Record actual SM count and residency limits
rather than inferring CTA waves from the generic H100 name.
[Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Measured Follow-Up: R64 Counters

The bounded counter probe now completed all six control/candidate launches on
an H100 with 132 SMs, CUDA 12.8 and Nsight Compute 2025.1. Each profiled producer
was isolated by its kernel name and an NVTX range after three warmup calls.
Both variants also passed FP32 output/LSE checks in the profiling workers.

The first attempt failed because the host denied GPU clock locking. Retrying
with `--clock-control none` collected counters successfully. These are
instrumented, unmanaged-clock profiles, not replacements for the earlier
uninstrumented paired latency measurements.

| B / N | Control instructions | Deferred instructions | Reduction | No eligible warp, control / deferred |
| --- | ---: | ---: | ---: | ---: |
| 1 / 4K | 7,163,904 | 7,139,328 | 0.34% | 76.21% / 76.49% |
| 1 / 16K | 12,306,432 | 12,183,552 | 1.00% | 72.64% / 73.13% |
| 2 / 4K | 14,327,808 | 14,278,656 | 0.34% | 71.74% / 72.09% |

All six launches used **164 registers/thread** and **65,536 dynamic shared
bytes/block**. Average eligible warps per scheduler remained only 0.27-0.36.
DRAM throughput was 3.92-11.44% of peak in these profiles. Nsight's stall-rule
report attributed roughly 37-41% of average warp cycles between issued
instructions to L1TEX long-scoreboard dependencies. This is an aggregate
diagnostic, not a source-PC attribution or a percentage of wall-clock runtime.

There is a useful arithmetic cross-check. Each four-lane row sum uses two
shuffle/add pairs. With two row fragments per lane and four warps per CTA,
deferring it from every tile to once per split removes approximately
`8 * (tiles_per_split - 1) * 4 * producer_CTAs` warp instructions. At these
three anchors this predicts 24,576, 122,880 and 49,152 fewer instructions,
exactly matching the measured total-count deltas. The resource allocation did
not change. This supports the intended operation removal, but it does not
explain the earlier 10% warm-graph gain or the B2 regression by itself.

**Next research question:** which instructions create those dependencies?
Collect source-correlated stalls before changing the producer. Distinguish the
synchronous scalar Q-staging loop from K/V load waits and the output epilogue.
If Q staging dominates, a bounded vectorized asynchronous Q-load replacement
is a smaller test than another whole-pipeline rewrite. If it does not, reject
that hypothesis. The present counters do not justify another R128 family,
register cap, split sweep or automatic promotion of deferred reduction.

- [Completed counter evidence](../artifacts/gate0/sm90_micro_counters_modal_h100_20260905_v2.json)
- [Clock-lock failure, retained](../artifacts/gate0/sm90_micro_counters_modal_h100_20260905.json)
- [Profiler command/source correlation options](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

## Global Sources: Useful Leads, Not New Claims

Zhejiang University's supercomputing-team lab material explains WGMMA
producer/consumer roles and explicitly warns that extra stages, producer warps,
and cluster geometry trade reuse against register/shared-memory occupancy.
It is a useful Chinese-language implementation reference, but its INT8/FP64
GEMM workload is not an attention benchmark.
[ZJUSCT HPC101 source](https://github.com/ZJUSCT/HPC101/blob/main/docs/lab/Lab4.5-INT8-FP64-GEMM/index.md).

The pipeline-parallel attention hardware description in US20240220572A1 was
screened as a dataflow lead. It does not provide an H100 CUDA implementation
or measured parity with our exact serving contract, so it is not evidence for
a promotion or a reason to change architecture now.
[Patent description](https://patents.google.com/patent/US20240220572A1/en).

## Where Deeper Work Is Still Needed

1. **Source-correlated latency attribution:** the first dynamic profiles point
   to low eligible-warp supply and long-scoreboard dependencies, with unchanged
   register/shared-memory allocation. Locate the responsible load instructions
   before replacing a pipeline. Static SASS alone is not a stall measurement.
2. **Exact mask lowering:** retain arbitrary logical positions as the oracle,
   then prove cheaper affine and further mask specializations equivalent.
   Direct paged/ragged FP16/BF16 execution now has independent H100 evidence.
3. **Whole-batch scheduling:** determine when one persistent mixed launch beats
   phase-separated launches after metadata and merge costs. Use frozen serving
   traces and holdouts, not an oracle's per-cell family choice.

These questions support the universal exact-engine plan. None calls for another
seed sweep, a return to Qwen-only policy tuning, or public promotion of the
rejected temporal/R128 families.
