# Adaptive Kernel: Goal and First Repair

## Objective

StreamAttn aims to avoid unnecessary attention work inside a native Triton/CUDA
streaming kernel. Exact online softmax is the accumulator and the zero-omission
control, not the distinguishing adaptive result. A fixed seed set or an offline
prompt-tier label is not the final query-dependent decision mechanism.

The intended two-stage decision is:

1. Before full K/V reads, use inexpensive cached information to justify omitting
   a block's contribution.
2. Otherwise evaluate QK, then decide whether V reads and PV are necessary.
3. Preserve the retained numerator/denominator and account for all omissions.
4. Evaluate more blocks whenever the decision cannot justify omission.

Existing exact serving defaults and calibrated policy contracts are unchanged.
This experiment does not make a generally verified model route available.

## The Accounting Defect

The old `certified_attention` reference and two-gate Triton prototype tested
each block against a fresh `error_budget`. That is not a total-output budget.
A deterministic regression uses one retained block and 127 blocks with relative
mass 0.001 each. Retained values are -1 and omitted values +1 on one coordinate.
Every individual omission costs about 0.002, but dropping the full tail costs:

```text
2 * 0.127 / (1 + 0.127) = 0.225377...
```

The old path accepted that tail with `error_budget=0.01`. Its final telemetry
reported the larger bound but did not prevent the overspend. Pre-only,
post-only, and combined-gate CPU regressions reproduced this before repair.

## Cumulative State

For retained support A and omitted support U:

```text
o_full - o_A = Z_U / (Z_A + Z_U) * (o_U - o_A)
```

Let `R` bound every legal value-vector norm, and let `U_bar` bound accumulated
omitted partition mass. All masses share the online accumulator's exponential
scale. Then, in real arithmetic:

```text
||o_full - o_A||_2 <= 2 R U_bar / (Z_A + U_bar).
```

A proposed block omission adds its mass bound to `U_bar`. The value-bound
predicate accepts only if the **new accumulated bound** fits the row budget.
When the running maximum changes, retained and omitted masses receive the same
rescaling. Subsequent evaluated blocks increase retained mass, so they cannot
invalidate this bound. `R` covers future as well as already evaluated values;
using only the current output norm here would need a different proof.

The separate `mass` predicate now bounds cumulative omitted/retained mass. It
does not promise an L2 output budget. `error_budget=0` disables omission.

This repair applies to `certified/attention.py` and
`kernels/certified_fwd_triton.py`. The separate legacy Gate-1 and projection
experiments retain their historical local-threshold semantics; they must not
be described as satisfying this cumulative contract.

This conservative rule repairs accounting; it does not solve the previously
measured looseness of centroid/radius summaries. It is not claimed as a new
selector or a formal floating-point certificate. FP32 summaries, softmax
arithmetic, FP16/BF16 PV and output rounding have separate numerical error.
Cached summaries must match current K/V contents. The reported LSE in the
reference is the retained-support normalizer, not full-context LSE after skips.

## Physical Work

The early two-gate Triton source masked scores but still loaded K/V and issued
matrix products. The repaired source branches around the load/MMA regions.
It supports compact GQA storage without expanding K/V, although the current
CTA owns one query head and does not yet reuse its loads across query heads.

Counters separate pre/post/computed row-blocks from valid CTA tiles, executed QK
tiles, and executed PV tiles. A mixed query tile can report many logical skips
yet still need every physical KV tile. These counters identify executed regions,
not HBM transactions or hardware instruction counts.

`materialize_skipped_work=True` deliberately runs masked work while preserving
the same mathematical selection. This diagnostic control tests whether branch
placement helps latency. It is never a serving-policy choice.

## GPU Canary

`benchmarks/profile_adaptive_two_gate.py` covers FP16/BF16, compact GQA,
cumulative tiny omissions, non-multiple lengths, top-left causal masks, peaked
rows, post-only decisions, and mixed rows sharing a query tile. It uses an
independent FP32 full-attention reference, a zero-budget control, a same-output
mask-only control, and graph replay after live query mutation.

Paired graph timings exclude counters and bound stores. Summary construction
is reported separately as wall time; no assumption about amortization turns it
into free work. A forced Flash SDPA baseline is attempted and failures retained.
This is a synthetic mechanism canary, not a fastest-baseline promotion, a
model-quality result, or an end-to-end inference speedup.

### H100 Results, 2026-09-09

Two Lightning submissions failed before GPU execution with `job reconciliation
failed`, each reporting zero cost. Both jobs were deleted. A bounded backup
canary and a precision diagnostic ran on Modal NVIDIA H100 80GB HBM3 with
PyTorch 2.7.1+cu128. Both apps stopped with zero tasks.

Ten of twelve canary cases completed the original checks. The two causal cases
failed the declared absolute row-L2 roundoff allowances; that failure is retained
in the [raw canary](../artifacts/gate0/adaptive_two_gate_modal_h100_20260909.json),
not changed into a pass. The [summary](../artifacts/gate0/adaptive_two_gate_summary_h100_20260909.json)
does not promote any route.

| Mechanism | Executed QK tiles | Executed PV tiles | Result |
|---|---:|---:|---|
| Peaked, both gates | 8 / 2,048 | 8 / 2,048 | Pre-K work elimination |
| Peaked, post-only | 2,048 / 2,048 | 8 / 2,048 | V/PV elimination only |
| Mixed rows in shared tiles | 2,048 / 2,048 | 2,048 / 2,048 | Logical skips saved no tile work |
| Cumulative tiny tail, four heads | 492 / 512 | 492 / 512 | Budget prevents blanket tail omission |

The FP16 cumulative case had row-L2 error `0.00805044`, with omission bound
`0.00889084` under the `0.01` budget. Query mutation replay passed for all ten
completed cases. Physical and mask-only controls produced identical output.

For the peaked FP16 case, median graph latency was `117.18 us` adaptive,
`369.99 us` mask-only, `218.06 us` zero-budget control, and `18.65 us` forced
Flash SDPA. The adaptive path therefore beat its own mask-only control but
still lost badly to the external exact baseline. Every timed canary case lost
to Flash SDPA; summary construction would add further cost unless amortized.
The serial scan and very small launch grid are plausible bottlenecks, not a
counter-proven stall attribution. Do not optimize this old execution shape
blindly or turn the synthetic skip fraction into a real-model result.

The [follow-up precision diagnostic](../artifacts/gate0/adaptive_two_gate_modal_h100_precision_20260909.json)
found zero omissions on the causal inputs and bit-identical adaptive/zero-budget
outputs. Maximum row-L2 errors were `0.00186488` FP16 and `0.01600127` BF16;
Flash SDPA had the same maxima. Casting the FP32 reference alone produced
`0.00186053` / `0.01595988`. Output representation accounts for almost all the
observed discrepancy. The original absolute limits are still failed. A future
evaluation must declare arithmetic/representation allowances separately from
the omission budget before measuring, rather than retroactively loosening it.

The kernel, reference, and summary-builder fingerprints match both GPU runs.
The profiler gained the optional causal diagnostic after the first canary, so
its full-file fingerprint differs in that earlier artifact. The raw hashes are
preserved rather than rewritten. Local validation finished with `1199 passed,
80 skipped`; CUDA/Triton tests skip on the Windows CPU environment. These GPU
mechanism runs do not replace model validation.

## What Research Still Has to Answer

- Can a cheap, value-aware decision recover useful omissions on held-out real
  activations after **cumulative** accounting, rather than local thresholds?
- Can those omissions eliminate shared physical GQA/page work without losing
  tensor-core utilization or paying more in decisions, metadata, and merging?
- Does the complete adaptive path preserve model behavior across prompts,
  horizons, layers, and model families while improving total inference time?

The next representation or predicate must change a demonstrated failure
mechanism. Another fixed-S sweep, replay of the old loose-summary configuration,
or exact-only layout win does not answer these questions. Exact page handling,
online state merging, and measured native kernels remain reusable assets.
