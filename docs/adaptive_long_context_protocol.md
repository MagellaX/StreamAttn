# Bounded 32K adaptive feasibility protocol

Predeclared before GPU execution. This is the final increase-context test of
the current shared hard-omission hypothesis, not a live selector or promotion.

## Frozen semantic contract

- Qwen/Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Ten fresh actual-model captures: layers 0/16/24/26/27, the existing technical
  and instruction_holdout_20260910 prompts, each tokenized to 32,768 positions.
- These prompts repeat short templates. They control the length comparison;
  they are not representative long-document traffic or new independent prompts.
- Dense upstream, final query N-1, post-RoPE, B1/M1/BF16, full G8 reuse,
  32-token logical blocks, cumulative absolute per-head L2 omission budget 1e-3.
- FP64 omission and native execution error are reported separately. The native
  component gate stays 0.02, not a native 1e-3 total-output certificate.

## Matched execution conditions

Each graph executes eight complete attention calls. Warm replay uses copy 0
eight times; rotating replay uses independent copies 0 through 7 once each.
All copies contain the same valid captured activations. No eviction kernel,
copy, preparation or gathering is timed. Events measure 100 graph replays;
reported milliseconds are divided by eight to give time per attention call.
Seven exact/selected pairs alternate order. Warm then rotating is a fixed
condition order, not an independent-worker or randomized-condition study.

The compact full KV payload is 32 MiB per copy and 256 MiB across eight copies.
Selected active KV payload is reported separately. Rotation is a controlled
working-set change, not a guarantee of cold L2, HBM traffic or production realism.
Hardware traffic counters are not required by this bounded run and absence is
recorded. [Hopper cache documentation](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)
and [FlashInfer rotating-buffer methodology](https://docs.flashinfer.ai/generated/flashinfer.testing.bench_gpu_time_with_cudagraph.html)
motivate the separation; the measured FlashInfer version stays pinned to 0.6.13.

Each condition independently resolves the fastest tested correct exact candidate,
including StreamAttn's full selected executor. Missing or incorrect candidates
remain explicit. A correct FlashInfer candidate is mandatory in both conditions.
All eight replicas are checked against FP64 before timing. Offline construction,
summary updates, selection and model forward are excluded from latency.

## Comparators and decisions

Report full control, current centered two-gate support, exact-mass-radius,
contribution-triangle, and hindsight best-found separately. All schedules are
full-information, supplied for free. Current two-gate support replay optimistically
skips known K as well as V; live post-QK decisions have already read K. Hindsight
exploits residual cancellation; a triangle certificate cannot. Neither is a
deployable selector. Identical schedules reuse timings and are not independent.

| Observation | Decision |
| --- | --- |
| Little shared omission across all searches | Stop tuning this formulation in the tested regime; not an impossibility theorem |
| Useful omission, negative headroom in both conditions | Attribute this adaptive executor before implementing another selector |
| Positive headroom only with hindsight cancellation | Investigate cheap cancellation identification, not selector integration |
| Positive conservative-comparator headroom | Measure minimum plausible decision and update cost |
| Complete adaptive margin on unseen inputs | Adaptive-conditioned generation and broader validation |

No automatic 64K escalation, budget relaxation, head-sharing reduction or dense
kernel redesign follows this run. Padding is a cost-attribution lead, not proof
of the latency bottleneck. Positive repetitive-input findings need a fresh
nonrepetitive holdout before any generalization claim.
