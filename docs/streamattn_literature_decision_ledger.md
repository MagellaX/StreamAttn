# StreamAttn Literature Decision Ledger

Date: 2026-06-04

This ledger maps the relevant long-context attention literature to concrete
StreamAttn route decisions. It is intentionally decision-oriented: papers are
benchmarks and evidence, not dependencies.

## Current Repo Evidence

The Qwen2.5-3B 32K/B8 validated route proves that StreamAttn can speed up
actual model decode, not just isolated attention calls:

```text
validated fast path v2, 32-step:
  dense decode:      28.47138 ms/token
  StreamAttn decode: 23.87325 ms/token
  speedup:           1.19261x
  top1/sample:       0 changes
  KL max:            9.95809e-05
```

The same route did not earn strict 128-step product promotion without verifier
help:

```text
validated fast path v2, 128-step:
  speedup:          1.16382x
  top1/sample:      0 changes
  KL max:           1.03894e-04
  top5 min:         3/5
  decision:         fail strict 128-step gate
```

The fp32 seed-reference backend reproduced the same margin failure class, so the
128-step issue is not primarily Triton online-softmax numeric drift.

Dynamic selection also improved stress coverage but did not promote stress
buckets:

```text
fixed_policy:
  support_out_seed_p95: 0.1734
  delta_collapse_p95:   0.1635
  value_residual_p95:   0.8423

qk_block_max:
  support_out_seed_p95: 0.1256
  delta_collapse_p95:   0.1140
  value_residual_p95:   0.6744

exact_mass_oracle:
  value_residual_p95:   0.6670
```

Decision:

```text
validated buckets:
  optimize product speed and verifier efficiency

stress buckets:
  do not promote fixed seed or dynamic seed alone
  use hybrid seed/dynamic/exact admissibility routing
```

## Paper Signals

### H2O / Scissorhands

Source:

```text
H2O: https://arxiv.org/abs/2306.14048
Scissorhands: https://arxiv.org/abs/2305.17118
```

Relevant signal:

```text
important-token persistence can reduce KV memory/work
recent tokens + historically important tokens are strong baselines
```

StreamAttn decision:

```text
fixed sink/recent/middle seeds are reasonable as Gate-0
but persistence alone is insufficient for adversarial stress buckets
```

Reason:

```text
stress failures remained after better selection, and high value residual
remained even under oracle-like selector diagnostics.
```

### Quest

Source:

```text
Quest: https://arxiv.org/abs/2406.10774
```

Relevant signal:

```text
query-aware page/block selection is the right shape for long-context decode
min/max page statistics approximate q-dependent criticality cheaply
```

StreamAttn decision:

```text
keep support-function dynamic selectors as a research branch
do not build a production selected-block kernel until replay passes safety
```

Reason:

```text
qk_block_max helped support coverage but did not remove stress KL/top-k/sample
risk for L26/L27. Better block choice is useful but not sufficient.
```

### PyramidKV / SqueezeAttention

Sources:

```text
PyramidKV: https://arxiv.org/abs/2406.02069
SqueezeAttention: https://arxiv.org/abs/2404.04793
```

Relevant signal:

```text
KV/cache budgets are layer-sensitive
uniform layer budgets are suboptimal
```

StreamAttn decision:

```text
route compiler must stay layer-specific
late layers need separate admissibility gates
```

Concrete policy implication:

```text
L24, L26, L27 should not share one global seed/dynamic decision on stress rows.
L27 should default exact for stress-risk buckets until a verifier proves safety.
```

### DuoAttention / Retrieval Heads

Source:

```text
DuoAttention: https://arxiv.org/abs/2410.10819
```

Relevant signal:

```text
some heads require retrieval/full context while others stream locally
```

StreamAttn decision:

```text
future policy should become layer/head aware
current route is layer-level because it is simpler and already measurable
```

Next research extension:

```text
head-level admissibility for L26/L27 after layer-level hybrid routes stabilize
```

### FlashAttention-3 / FlashAttention-4

Sources:

```text
FA3: https://arxiv.org/abs/2407.08608
FA4: https://arxiv.org/abs/2603.05451
```

Relevant signal:

```text
exact attention wins through hardware-specific dataflow, async pipelines,
softmax/matmul overlap, and Blackwell-specific non-matmul reduction
```

StreamAttn decision:

```text
do not try to be a worse dense exact kernel
win by scheduling much less work and by owning the model integration path
```

Validated-bucket backend target:

```text
native routed Qwen module
projection path reduction
RoPE/cache append/seed fusion
eventual exact_native verifier
```

### FlashMLA-ETAP / SnapMLA

Sources:

```text
FlashMLA-ETAP: https://arxiv.org/abs/2506.01969
SnapMLA:       https://arxiv.org/abs/2602.10718
```

Relevant signal:

```text
high-performing decode systems rewrite dataflow, not just masks
RoPE/KV layouts and quantization sensitivity matter
```

StreamAttn decision:

```text
packed/native cache and fused routed-module paths are higher reward than
another tiny seed-kernel tweak
```

Future quantization rule:

```text
test seed-cache FP8/FP4 only under logit replay and closed-loop gates
keep RoPE-sensitive components high precision until proven safe
```

### RULER / NoLiMa

Sources:

```text
RULER:  https://arxiv.org/abs/2404.06654
NoLiMa: https://arxiv.org/abs/2502.05167
```

Relevant signal:

```text
needle-in-haystack alone is too weak
literal matching overestimates real long-context robustness
```

StreamAttn decision:

```text
strict promotion must include adversarial stress buckets, long-horizon replay,
and distribution-level gates
```

Current status:

```text
Qwen3B validated route is Gate-0 actual-model proof
not strict long-horizon product promoted without verifier assistance
```

### Flux Attention / SpecBound

Sources:

```text
Flux Attention: https://arxiv.org/abs/2604.07394
SpecBound:      https://arxiv.org/abs/2604.12247
```

Relevant signal:

```text
dynamic routing should be context-aware
confidence/difficulty calibration can preserve correctness while saving work
```

StreamAttn decision:

```text
build a late-layer admissibility router:
  seed-only if safe
  dynamic seed if coverage-repairable
  exact if low-margin/value-sensitive
```

The exact verifier should be event-triggered and row/layer-selective, not
periodic:

```text
risk = low_logit_margin
     + value_residual_proxy
     + attention_entropy
     + bucket_risk
     + layer_risk

if risk < seed_threshold:
    seed_only
elif risk < dynamic_threshold:
    dynamic_seed
else:
    exact_native
```

## StreamAttn-Specific Moat

The key distinction from generic sparse attention is:

```text
StreamAttn does not optimize attention-matrix approximation.
StreamAttn optimizes model-output-preserving attention sufficiency.
```

The current route formula remains:

```text
choose cheapest native mode A
subject to D_model(P_exact, P_A) <= epsilon
and occupancy/backend constraints
```

For head-private seed-only:

```text
bytes_ratio = G * S / N
CTA_count   = B * Hq * Csplit
```

But stress evidence adds a missing gate:

```text
value_residual and late-layer composition risk must be admissible
```

So the route compiler objective is now:

```text
maximize full-model speed
subject to:
  top1/sample stable
  KL gate passes
  top-k retained probability mass acceptable
  stress-risk buckets exact or verifier-safe
```

## Immediate Next Route

Run the hybrid stress route matrix generated by:

```text
benchmarks/plan_hybrid_stress_routes.py
```

Priority order:

```text
1. stress_l27_exact_l26_seed
2. stress_l27_exact_l26_dynamic_extreme4
3. stress_l27_exact_l26_dynamic_qk
4. stress_l26_l27_exact
5. stress_l24_l26_l27_exact
```

Expected decision:

```text
if L27 exact + L26 seed/dynamic passes:
  package stress hybrid route

elif only L26/L27 exact passes:
  stress buckets exact for L26/L27

elif late exact still fails:
  stress buckets exact_native until trained/router adaptation
```

## Token / Modal Rule

HF tokens must be used only through environment variables or Modal secrets.
Do not paste token values into commands, source files, artifacts, or docs.

Expected secret setup:

```text
modal secret create huggingface-token HF_TOKEN=<read-only token>
set STREAMATTN_MODAL_HF_SECRET=huggingface-token
```

