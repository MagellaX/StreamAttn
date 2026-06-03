# Qwen2.5-3B Dynamic Selector Findings

Date: 2026-06-04

## Scope

Risk-plan-driven selector coverage was run on the Qwen2.5-3B 32K/B8 strict stress failures.

Target:

```text
layers: 24, 26, 27
route:  [0, 14, 16, 24, 26, 27, 35]
steps:  0, 1, 2, 3, 5, 11, 16, 17, 18, 21
rows:   compiled stress risk rows
```

Selectors:

```text
fixed_policy
support_top2_norm_refine32
support_top4_norm_refine32
support_extreme4_mean_refine32
support_rand8_refine32
qk_block_max
exact_mass_oracle
value_residual_oracle
```

## Result

Dynamic selection improves prompt-aware coverage versus the fixed S384 policy, but does not make the stress route product-promotable.

Selector-level p95 metrics:

```text
fixed_policy:
  mass_omitted_p95:        0.7093
  support_out_seed_p95:    0.1734
  delta_collapse_p95:      0.1635
  value_residual_p95:      0.8423

qk_block_max:
  mass_omitted_p95:        0.5193
  support_out_seed_p95:    0.1256
  delta_collapse_p95:      0.1140
  value_residual_p95:      0.6744

support_extreme4_mean_refine32:
  mass_omitted_p95:        0.5545
  support_out_seed_p95:    0.1262
  delta_collapse_p95:      0.1152
  value_residual_p95:      0.7188

exact_mass_oracle:
  mass_omitted_p95:        0.5095
  support_out_seed_p95:    0.1319
  delta_collapse_p95:      0.1193
  value_residual_p95:      0.6670
```

Relative to fixed policy:

```text
qk_block_max support_out p95 improves by ~27.5%
qk_block_max delta_collapse p95 improves by ~30.3%
support_extreme4_mean_refine32 is close to qk_block_max
```

## Interpretation

The query-aware selector direction is real: support-function proxies recover meaningful coverage without scanning the full prefix.

However, even qk_block_max and oracle selectors still leave high residual stress risk:

```text
qk_block_max support_out_seed_p95: 0.1256
qk_block_max value_residual_p95:   0.6744
exact_mass_oracle value_residual:  0.6670
```

That means fixed seed is not the only blocker. Stress buckets also expose late-layer value sensitivity and composition sensitivity.

Layer risk:

```text
L24:
  mass_omitted_p95:     0.3097
  value_residual_p95:   0.4565

L26:
  mass_omitted_p95:     0.4711
  value_residual_p95:   0.6927

L27:
  mass_omitted_p95:     0.7092
  value_residual_p95:   0.8180
```

L27 remains the hardest late layer. L26 is also risky. L24 is secondary.

Bucket risk:

```text
chat_instruction:
  support_out_seed_p95: 0.2579
  value_residual_p95:   0.8327

json_tool:
  support_out_seed_p95: 0.2765
  value_residual_p95:   0.8131

needle_rag:
  support_out_seed_p95: 0.1537
  value_residual_p95:   0.7274
```

Code and long_doc look much less risky by coverage metrics.

## Decision

Do not promote dynamic selection for stress buckets yet.

Use:

```text
validated buckets:
  keep optimizing seed-only product speed

stress-risk buckets:
  keep L26/L27 exact or bucket-gated
  use dynamic selector as research evidence, not product policy
```

Next high-reward work is back to validated-bucket product speed:

```text
native routed Qwen module
projection path reduction
RoPE + cache append + seed attention fusion
```

The stress-bucket research branch should continue only if we test stronger interventions:

```text
query-aware selector + exact late layer
L27 exact with L24/L26 dynamic
confidence-margin gate
trained StreamAttn-native adaptation
```
