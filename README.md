# StreamAttn

[![CI](https://github.com/MagellaX/StreamAttn/actions/workflows/ci.yml/badge.svg)](https://github.com/MagellaX/StreamAttn/actions/workflows/ci.yml)
[![CUDA Source Build](https://github.com/MagellaX/StreamAttn/actions/workflows/gpu-source.yml/badge.svg)](https://github.com/MagellaX/StreamAttn/actions/workflows/gpu-source.yml)

**A native attention engine for exact streaming attention and model-validated
reduced-work decode.**

StreamAttn owns its attention kernels. Its default native serving path does not
call FlashInfer or FlashAttention; those projects are comparison baselines. At
runtime, StreamAttn can compute exact attention over the full KV cache or, for
an explicitly calibrated model/shape cell, execute a much smaller seed-only
schedule and fail closed to exact attention when the cell does not match.

The common foundation is single-pass streaming attention with online softmax:
K/V tiles are consumed once, numerically stable running statistics are updated
on the fly, and the full attention matrix is never materialized.

> **Project status:** research engine with promoted H100 routes, guarded
> contiguous A100 D64 routes plus one cross-provider D128 cell, a complete A100
> paged-D128 routing profile, and partial H100/B200 phase databases. StreamAttn
> has apples-to-apples exact decode wins over FlashInfer on promoted Hopper and
> Ampere cells, plus a measured model-level Qwen decode win on a validated
> request tier. The A100 paged-D128 profile still selects exact external
> fallback. StreamAttn is not yet a universal replacement for FlashInfer,
> FlashAttention, or a full serving runtime.

## Why StreamAttn Exists

Fast exact kernels answer this question:

```text
How efficiently can the GPU compute all requested attention work?
```

StreamAttn also asks:

```text
What is the cheapest native attention route that is valid for this request?
```

That produces three serving modes plus an explicit adaptive research route:

| Mode | Work performed | Semantics | Current use |
|---|---|---|---|
| `exact_native` | All KV tokens | Exact attention, within numerical tolerance | Default and fail-closed route |
| `seed_only_native` | A small sink/middle/recent seed set | Approximate; only for a packaged, validated policy cell | Explicit opt-in and calibrated serving |
| `verified_auto` | Seed-only when policy invariants match, otherwise exact | Policy-verified, fail-closed routing | Default planning mode; live generic verification is still research |
| query-selected paged | Sink/recent atoms plus query-ranked middle atoms | Exact online softmax over a runtime-selected subset | Explicit `StreamAttnEngine.plan_query_selected_paged(...)` research route |

StreamAttn is therefore not just another sparse mask. The engine owns the
kernel, policy artifact, fixed-buffer plan, route decision, and exact fallback.

### One engine, multiple tile schedules

The engine is organized around the attention work, not around one model or one
kernel. Every `StreamAttnEngine` plan now lowers through three explicit layers:

```text
AttentionProblem
  semantic guarantee, Q/KV geometry, dtype, mask, cache kind

AttentionTilePlan
  logical KV source + the tiles that are legal to execute

AttentionBackendPlan
  architecture-specific kernel, splits, and workspace
```

This gives exact and reduced-work attention one semantic contract:

```text
exact contiguous  -> all logical tiles, contiguous mapping
exact paged       -> all logical tiles, page-table mapping
validated fixed   -> calibrated logical tile subset
adaptive          -> runtime-selected logical tile subset (research)
sliding window    -> bounded logical tile range (planned)
```

The runtime executes contiguous selected schedules and contiguous or paged
exact schedules. Selected paged work compiles into a device CSR schedule and
page-native `PackedRoute64` metadata without copying K/V. H100 now executes
those records directly with the same transposed WGMMA and online-softmax
mainloop used by exact paged decode. The same planner now describes exact
decode, prefill, and differentiable training calls. Future adaptive,
compressed, and device backends therefore do not need a second semantic API
or a second online-softmax model. See [the universal tile
planner](docs/universal_attention_tile_planner.md) and [the selected paged
route ABI](docs/selected_paged_route_abi.md).

## What Is Proven Today

### Exact native decode

The promoted SM90 kernels compute full-context GQA decode and use online
softmax plus exact split-state merging. Contiguous and page-64 results below
use FlashInfer 0.6.12; page-16 was re-gated against FlashInfer 0.6.17 on H100:

| Shape family | Measured region | StreamAttn speedup | Status |
|---|---|---:|---|
| Contiguous D64, GQA group 8 | A100 40GB; B1-B8; 16K-64K KV | `1.067x-1.477x` | Promoted per cell |
| Contiguous D64, GQA group 4 | A100 40GB; B4/32K KV | `1.070x` | Promoted per cell |
| Contiguous D128, GQA group 8 | A100 SXM4 40GB/80GB; B4/16K KV | `1.006x` worst paired extension trial; `1.010x-1.014x` production-plan medians | Promoted discrete cell |
| D64, GQA group 8 | 7 cells; B2-B8; 16K-64K KV | `1.025x-1.432x` | Promoted per cell |
| D64, GQA group 4 | 14 cells; B1-B16; 16K-64K KV | `1.027x-1.449x` | Promoted per cell |
| D128, GQA group 4 | 6 cells; B4-B16; 16K-64K KV | `1.002x-1.012x` | Promoted per cell |
| Paged D64, GQA group 8 | HND/page-64; B1-B8; 16K-64K KV | `1.21x-2.24x` paired median | Promoted per cell |
| Paged D64, GQA group 8 | HND/page-16 full rows; B1-B8; 16K-64K capacity | `1.21x-2.07x` paired median | Promoted per cell |
| Paged D64, GQA group 8 | HND/page-16 ragged rows; same capacity matrix | `1.17x` worst paired trial; `2.04x` median cell | Promoted per cell |
| Paged D128, GQA group 8 | HND/page-16; B1-B8; 16K-64K; full and ragged | `1.075x` worst paired trial; `1.75x` median auto-gate cell | Promoted per cell |
| Paged D128, GQA group 8 | NHD/page-16; B1-B8; 16K-64K; full and ragged | `1.058x` worst paired trial; `1.283x` median auto-gate cell | Promoted per cell |
| Paged D128, GQA group 4 | HND/page-16; B1-B8; 16K-64K; full and ragged | `1.017x` worst paired trial; `1.34x` median cell | Promoted per cell |
| Paged D128, GQA group 8 | A100/HND and NHD/page-16; B1-B8, 16K-64K exploratory matrix | `6/12` discovery-process wins per layout; strict B1/32K/HND `0.949x` | Segmented exact merge is correct; fast-SXM route remains external fallback |
| Paged D128, GQA group 8 | B200/NHD/page-16; six B1-B8, 32K/64K full-row cells | `1.144x-1.441x` initial confirmation; `1.122x` worst paired trial | Promoted per cell |

The exact backend uses transposed WGMMA so that long context occupies the
hardware-friendly matrix dimension:

```text
context tokens  -> WGMMA M
query heads     -> WGMMA N
head dimension  -> WGMMA K reduction
```

See the measured phase diagrams for the actual promoted cells:

- [D64/G8 H100 phase diagram](docs/exact_native_h100_phase_diagram_20260818.md)
- [D64/G4 H100 phase diagram](docs/exact_native_h100_g4_phase_diagram_20260818.md)
- [D128/G4 H100 phase diagram](docs/exact_native_h100_d128_g4_phase_diagram_20260819.md)
- [Paged D64/D128 H100 phase diagram](docs/paged_exact_decode.md)
- [Paged exact SM80/SM90/SM100 architecture phase](docs/paged_exact_architecture_phase_20260824.md)
- [SM80 segmented exact merge and falsification boundary](docs/sm80_segmented_exact_merge_20260829.md)
- [Contiguous D64 A100 phase diagram](docs/exact_native_a100_d64_phase_diagram_20260831.md)
- [Contiguous D128 A100 register-resident pipeline](docs/exact_native_a100_d128_register_pipeline_20260901.md)
- [Native B200 exact backend and promotion boundary](docs/paged_exact_sm100_tgv_20260824.md)

The promoted A100 D64 path is a separate architecture-native family. It reads
standard `[B,Hq,D]` queries and writes `[B,Hq,D]` output directly, stages K/V
with `cp.async`, assigns four producer warps per CTA, and merges exact online-
softmax states hierarchically. Its production plan preallocates all workspace;
the B4/G8/32K allocation-free canary measured `63.155 us` against FlashInfer
0.6.13 at `73.020 us`, with both paths at `1.22e-4` BF16 max error. The
ThunderKittens headers are optional; set `STREAMATTN_TK_ROOT` to a tree
containing `include/kittens.cuh`, otherwise the dispatcher fails closed to an
existing exact path.

The promoted A100 D128 cell uses a different shared-memory lifetime. Four
producer warps load each current K/V pair into registers, asynchronously replace
the shared slots with the next pair while QK, online softmax, and PV execute, and
reuse K storage for the final partial output. The real `ExactDecodePlan.run()`
path passed 15/15 paired trials at B4/G8/16K on both A100 SXM4 40GB and 80GB.
All other A100 D128 cells still fail closed to exact fallback.

### Selected paged decode

The H100 selected-paged backend consumes `PackedRoute64` records directly from
page-16 NHD or HND caches. One producer CTA evaluates one selected 64-token
record, applies per-page Q-head and token masks before softmax, and emits an
exact online-softmax partial state. A static row merge produces the final
output. K/V data is never gathered or repacked.

Measured against the fastest tested FlashInfer exact decode backend at
NHD/page-16, BF16, D128/G8, 32K:

| Route | B1 | B4 | B8 | Paired evidence |
|---|---:|---:|---:|---|
| 384 selected tokens | `2.67x` | `5.16x` | `8.25x` | Independent confirmation; `45/45` wins |
| 2,048 selected tokens | `2.76x` | `5.08x` | `7.18x` | Independent confirmation; `45/45` wins |
| 8,192 selected tokens | `2.06x` | `2.84x` | `2.32x` | Initial phase; `21/21` wins |
| 16,384 selected tokens | `1.69x` | `1.36x` | `1.36x` | Initial phase; `21/21` wins |
| Full 32K control | unstable | `0.78x` | `0.75x` | Route to the exact split scheduler |

All 15 phase cells matched an independent FP32 selected-token reference. A
Q-head-private 384-token schedule with only `0.545` GQA union efficiency also
won all `27/27` paired trials (`2.44x`, `5.21x`, and `8.20x` at B1/B4/B8).
These are kernel/runtime results for a precomputed selected schedule.

StreamAttn also has a no-sync dynamic route compiler for mutable GPU Q-head
CSR atoms. It builds a bounded on-chip atom/head membership map, compacts the
GQA union with warp ballots, resolves live page IDs, and launches the selected
WGMMA executor without a host route-count readback. At the same 32K D128/G8
shape, route preparation fell from `1.532-3.003 ms` in the generic Torch
lowering to `0.01098-0.01362 ms` on GPU.

Including dynamic preparation and execution, all 18 B1/B4/B8 x S384/S2048 x
shared/alternating/disjoint cells were correct and won paired per-call trials
against FlashInfer exact (`1.112x-5.276x`). A 400-call amortized measurement
won 15/18 cells; every S384 cell won, while three low-batch S2048 corners remain
exact-fallback territory. See the [dynamic H100 route-compiler
phase](docs/paged_dynamic_selected_h100_phase_20260826.md).

The next phase connects a real GPU query selector directly to that mutable CSR.
Persistent per-64-token support keys are scored against the live query; a
fixed-width top-k kernel emits score-ranked Q-head atoms; the membership
compiler canonicalizes their GQA union without a sort or host readback. The
complete measurement includes selection, route lowering, page resolution,
attention, and merge:

| Support sketch | B1 | B4 | B8 | Selection-quality signal |
|---|---:|---:|---:|---|
| P1/P2/P4 centroid + extremes, S384 | `0.55x-0.57x` | `1.12x-1.14x` | `1.69x-1.70x` | P4 synthetic block-max recall: `15%-21%` |
| P8 centroid + extremes, S384 | not promoted | `0.997x` | `1.535x` | synthetic block-max recall: `33%-36%` |

All 13 measured cells matched the independently lowered selected-token
reference. Every P1/P2/P4 B4/B8 cell won the seven paired block-timed trials;
B1 lost because the roughly `0.05 ms` selector floor dominates. P8 buys more
oracle recall but consumes the B4 margin, so adaptive sketch width is required
rather than one global selector. Top-norm support keys were measured and
rejected: they did not improve recall and were slower. See the [query-selected
H100 phase](docs/paged_query_selected_h100_phase_20260826.md).

The runtime also supports a two-stage selector:

```text
P4 support scan -> top-32 candidates -> exact 64-token block-max QK
-> final four middle atoms -> selected WGMMA attention
```

This path scans compact P4 metadata over 32K and exactly refines only 2,048
candidate tokens per Q head. At B8, an independent 15-trial confirmation won
`15/15` paired trials against FlashInfer exact (`1.056x` median, `1.029x`
worst), but the preceding phase won only `8/9` (`0.985x` worst). It therefore
remains a narrow experimental boundary. At B16, refine-8/16/32 all won `9/9`;
refine-32 reached `1.619x` paired median and `1.544x` worst. All eight phase
cells matched the selected-token reference. Exact refinement improves the
ranking inside the proxy candidate set; it does not make the selected route
equivalent to full-context exact attention.

These query-selected results are systems evidence, not a model-safety
promotion. Attention is exact over the selected atoms, but arbitrary runtime
selection can change model outputs. Existing real-Qwen analysis shows that
support sketches improve coverage while adversarial late-layer routes still
need exact fallback or a stronger verifier. The measured cache is fixed at
32K; a growing production cache must finalize/update the support metadata when
each new 64-token atom closes.

### Adaptive output-sufficiency research

StreamAttn now has a reference frontier for a harder question: can selected
exact blocks plus a compact tail estimator preserve the post-`o_proj` output
without scanning the omitted KV tail? It merges exact online-softmax block
states with moment estimates or sampled control-variate corrections and keeps
physical routes shared within each true-GQA KV group.

Initial H100 captures were deliberately negative-to-mixed. Output-aware block
selection improves hard drop, while moment completion, fixed residual banks,
and stochastic ratio correction do not generalize reliably. The follow-up
predictability gate now uses the exact factorization
`o = o_A + sigmoid(log(Z_U/Z_A)) * (o_U - o_A)`. Omitted normalization mass is
highly predictable, but the vector-valued innovation is not reliably
low-rank across prompts. Qwen L14/L26/L27 failed promotion. Mistral L0 retained
a `30.23%` mean unseen-prompt error reduction, but its worst p95 row regressed
by `12.6%`. A nested exact-canary follow-up ranked those failures from only
runtime-observable features: it accepted `18/64` unseen-prompt rows
(`28.125%`), accepted zero regressions, and achieved `0.942` risk AUC. The
worst accepted row still improved over hard drop (`0.975x` error). This is a
semantic canary pass, not a backend promotion: selection still uses the exact
QK oracle and predictor/summary maintenance cost is not yet positive. See the [frontier
study](docs/adaptive_output_sufficiency_frontier_20260827.md) and the
[conditional predictability gate](docs/adaptive_residual_predictability_20260827.md).

### Model-aware reduced-work decode

For calibrated Qwen-family cells, the seed-only backend reads a small set of
sink, middle, and recent KV blocks. A representative 32K policy uses 384 seed
tokens, or `1.17%` of the full context. In true GQA, duplicating those reads
across query heads is still inexpensive when:

```text
G * seed_tokens / kv_len << 1
```

For the original Qwen route, that ratio is about `8.2%` of exact KV traffic.
The extra head-private work creates enough independent CTAs to occupy the GPU
while avoiding most of the exact QK, softmax, and PV work.

Measured complete-model results on H100:

| Route | Scope | Speedup | Distribution result | Status |
|---|---|---:|---|---|
| Qwen2.5-3B fast path | 32K, B8, validated request suite, 32 decode steps | `1.193x` | Zero top-1/sample changes; KL max `9.96e-05` | Candidate |
| Qwen2.5-3B `verified_auto` | 32K, B8, validated suite, 128 decode steps | `1.157x` | Zero top-1/sample changes; strict gates pass | Research candidate; offline canary |

These rows compare complete model decode against the dense Hugging Face model,
not just selected attention calls. Adversarial stress and unknown request tiers
remain exact. The 128-step verifier currently uses an offline-calibrated
trigger schedule, so it is evidence for the design rather than a general live
verifier.

### What the numbers do not claim

- StreamAttn is not universally faster than FlashInfer on every exact shape.
- Seed-only speedups are not exact-kernel speedups; they come from validated
  work avoidance.
- A100 promotion is limited to contiguous head-major BF16 D64 G4/G8 cells and
  the discrete contiguous D128 B4/G8/16K cell. Tested D128 32K, G4, and B8 cells
  remain on exact fallback. Native HND/NHD page-16 D128/G8 is correct, but its
  strict HND gate resolves
  to `0.060416 ms` versus a `0.057344 ms` FlashInfer FA2 floor, so the complete
  seven-cell paged-D128 database still emits exact external fallbacks. No A100
  FP8 claim has been promoted.
- B200 promotion is narrow: BF16, direct NHD page-16, D128/G8, full fixed rows,
  and six measured B/N cells only. B1/64K and B8/64K did not clear the paired
  gate and remain on exact fallback. This is not a universal Blackwell claim.
- H100 paged exact promotion covers measured
  D64/G8 and D128/G4/G8 HND cells plus D128/G8 NHD cells at 16K/32K/64K.
  Page-16 supports exact variable-length rows; page-64 is D64/G8-only and
  still requires full fixed-length rows.
- The repository does not currently claim a direct universal win over
  FlashAttention training kernels.

## Installation

StreamAttn requires Python 3.10+ and PyTorch. Triton is optional for CPU use but
required for the native CUDA paths.

```bash
git clone https://github.com/MagellaX/StreamAttn.git
cd StreamAttn

# Development install
python -m pip install -e ".[dev]"

# Native Triton kernels on supported Linux/x86-64 systems
python -m pip install -e ".[triton]"

# Optional Hugging Face helpers
python -m pip install -e ".[hf]"
```

The package is currently a portable `py3-none-any` wheel. GPU kernels are
compiled for the installed device at runtime, so separate A100/H100/B200 Python
wheels are not needed.

## Quick Start

### One-shot exact decode

Decode tensors use `[batch, sequence, heads, head_dim]`. For autoregressive
decode the query sequence is normally one token.

```python
import torch
import stream_attention as stream_attn

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

q = torch.randn(4, 1, 16, 64, device=device, dtype=dtype)
k_cache = torch.randn(4, 32768, 2, 64, device=device, dtype=dtype)
v_cache = torch.randn_like(k_cache)

output, info = stream_attn.decode(
    q,
    k_cache,
    v_cache,
    mode="exact_native",
)

print(output.shape)
print(info.backend_used)
```

On an unsupported native CUDA shape, the exact path uses the safe exact
implementation available in the installed environment. Promoted serving cells
are always explicit; StreamAttn does not infer a performance claim from a
successful launch.

### Exact decode from a paged KV cache

Paged decode accepts physical NHD or HND pages and a per-request block table.
The native kernels resolve logical tokens to physical pages inside the
online-softmax loop; they do not gather pages into a contiguous cache. The
promoted H100 page-16 specializations accept HND, plus direct NHD for D128/G8.
Page-64 maps HND directly to one transposed-GQA WGMMA tile. Page-16 loads four
physical pages directly into the same 64-token shared tile without a gather
buffer; NHD changes only the producer's token stride, not the WGMMA consumer.

~~~python
import torch
import stream_attention as stream_attn

q = torch.randn(4, 1, 16, 64, device="cuda", dtype=torch.bfloat16)
k_pages = torch.randn(8192, 2, 16, 64, device="cuda", dtype=torch.bfloat16)
v_pages = torch.randn_like(k_pages)
page_table = torch.arange(
    8192, device="cuda", dtype=torch.int32
).view(4, 2048)
sequence_lengths = torch.full(
    (4,), 32768, device="cuda", dtype=torch.int32
)

cache = stream_attn.PagedKVCache(
    key=k_pages,
    value=v_pages,
    page_table=page_table,
    sequence_lengths=sequence_lengths,
    layout="HND",
)

plan = stream_attn.StreamAttnEngine().plan(
    q,
    cache,
    mode="exact_native",
)
output, info = plan.run()
~~~

The page table may be updated in place between steps while preserving its
validated shape and bounds. The promoted page-16 WGMMA plan accepts positive
per-request lengths, masks the final 64-token compute tile exactly, and ignores
inactive `-1` table slots. Page-64 WGMMA still requires full planned lengths;
unsupported shapes use the generic exact paged backend. Both paths are
allocation-free after planning.

The same page-16 API also promotes D128 BF16 cells with GQA group sizes four
and eight. D128 reuses each shared-memory stage for V only after its K tile has
been consumed, keeping the two-phase pipeline inside Hopper's shared-memory
budget without gathering pages.

### Plan once for a decode loop

Stable serving loops should validate shapes and policy once, then reuse the
same Q/K/V and output buffers:

```python
import torch
from stream_attention import StreamAttnEngine

q = torch.randn(4, 1, 14, 64, device="cuda", dtype=torch.float16)
k_cache = torch.randn(4, 32768, 2, 64, device="cuda", dtype=torch.float16)
v_cache = torch.randn_like(k_cache)

engine = StreamAttnEngine(
    policy_name="qwen25_05b_l8_32k_seed_only_batched",
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    layer_id=8,
)

plan = engine.plan(
    q,
    k_cache,
    v_cache,
    mode="verified_auto",
)

# Update q/k_cache/v_cache contents in place between decode steps.
output, info = plan.run()
print(info.backend_used, plan.reason)
```

`verified_auto` selects seed-only only when model, layer, dtype, batch, KV
bucket, GQA shape, device, and policy constraints match. Otherwise it executes
exact attention. `seed_only_native` is the strict explicit mode and raises when
the requested seed route cannot be planned.

### Prefill and training attention

The public engine surface covers all three attention phases. Prefill and
training use exact all-tile schedules and the fused online-softmax
forward/backward when the native Triton path supports the call. Unsupported
devices or features use PyTorch SDPA without changing the public tensor layout:

```python
import torch
import stream_attention as stream_attn

q = torch.randn(2, 1024, 8, 64, device="cuda", dtype=torch.float16,
                requires_grad=True)
k = torch.randn(2, 1024, 2, 64, device="cuda", dtype=torch.float16,
                requires_grad=True)
v = torch.randn_like(k)

prefill_output, prefill_info = stream_attn.prefill(
    q.detach(), k.detach(), v.detach(), causal=True, return_info=True
)
output = stream_attn.train(q, k, v, causal=True)
output.square().mean().backward()
```

Both calls use `[batch, sequence, heads, head_dim]` tensors and lower through
`AttentionProblem -> AttentionTilePlan -> AttentionBackendPlan`. The native
CUDA path accepts compact GQA K/V tensors: each query head maps to its KV head
inside the forward and backward kernels, so native execution does not
materialize repeated KV heads. CPU and unsupported-shape fallback explicitly
report `torch_sdpa_gqa_expanded` when they must expand GQA for SDPA.

The underlying Triton path supports boolean/additive masks, deterministic
Philox dropout, ALiBi, and a streaming backward pass. Training calls with a
feature combination that is not natively lowered use the exact SDPA fallback.
`stream_attn.train(...)` computes differentiable attention; optimizer and model
training-loop ownership remain with the caller. H100/B200 correctness,
backward derivation, and the first performance phase diagram are recorded in
[unified prefill and training
attention](docs/functional_attention_api_20260828.md). The compact lowering is
a correctness/generalization milestone, not yet a promoted performance path:
its Q-head-owned programs reload shared K/V for every GQA query head and its
backward atomically merges those contributions. FlashAttention-class SDPA is
still faster across the measured prefill/training cells.

An experimental KV-group-owned forward floor now flattens multiple query heads
into one program's row tile and reuses compact K/V while maintaining independent
FP32 online-softmax states. It improves the first native forward by up to
`4.85x` on H100 and `2.72x` on B200, but its best measured paired result is
still `0.89x` of Flash SDPA. It is therefore not auto-routed. See the
[grouped-GQA prefill floor](docs/grouped_gqa_prefill_floor_20260828.md).

A native Hopper follow-up replaced the Triton dot-product floor with natural
`m64n64` WGMMA and tested both one and two consumer warpgroups sharing K/V.
All 12 H100 G4/G8 D128 cells were exact, but the two-consumer schedule reached
only `0.408x-0.551x` of graph-captured Flash SDPA and was limited to one CTA/SM
by 223 registers per thread and 85,120 bytes of dynamic shared memory. It is a
recorded negative canary, not a production route. See the
[SM90 grouped WGMMA study](docs/sm90_grouped_gqa_prefill_wgmma_20260902.md).

The execution-state follow-up found one reusable mechanism and closed three
topologies. Register/shared PV improved the serial attention epoch by
`1.177x-1.494x` over shared/shared PV. A fair producer-consumer TMA epoch,
however, reached only `0.810x-0.962x` of the lean vectorized serial path; a
same-CTA two-consumer version spilled and reached `0.723x`. Two-CTA TMA
multicast preserved `0.948x-0.978x` of independent load throughput, but its
complete attention epoch still reached only `0.522x-0.712x` of serial RS.
StreamAttn therefore retains RS-PV and the multicast transport primitive while
rejecting these producer-heavy attention schedules. See the [SM90 RS-PV
execution-state study](docs/sm90_grouped_prefill_rs_pv_epoch_floor_20260902.md).

The resulting complete 128-thread consumer-owned kernel is the first positive
Hopper grouped-prefill result. It combines double-buffered `cp.async` K loads,
shared/shared QK WGMMA, exact FP32 online softmax, register/shared PV WGMMA, and
direct output/LSE writes. The compiled kernel uses 168 registers per thread,
65,536 bytes of dynamic shared memory, zero local memory, and allows three
CTAs/SM. All 30 tested H100 B1/B2/B4, G4/G8, S512-S8192 cells passed output and
sampled-LSE correctness. Eleven cells passed the strict paired performance gate:
B1 at G4 S4K/S8K and G8 S2K/S4K/S8K; B2 at G4/G8 S4K/S8K; and B4 at G4/G8
S8K. Their median speedups over graph-captured Flash SDPA range from `1.02x` to
`1.18x`, while the new dataflow is roughly `1.69x-2.15x` faster than the prior
complete shared/shared PV canary. These are phase-compiler candidates, not yet
public auto-routes; short and medium losing cells remain on exact fallback. See
[SM90 consumer-owned RS-PV grouped
prefill](docs/sm90_grouped_rs_prefill_complete_20260902.md).

For Blackwell, a separate architecture-native exact forward keeps compact GQA
K/V in BSHD layout and uses TMA,
`tcgen05` MMA, TMEM accumulators, streaming online softmax, query-tile causal
truncation, and row masking on the diagonal tile. The original eager-only
comparison against forced PyTorch Flash SDPA measured:

| Batch / sequence | StreamAttn | Flash SDPA | Paired speedup |
|---|---:|---:|---:|
| B1 / S64 | `0.00619 ms` | `0.01020 ms` | `1.65x` |
| B1 / S128 | `0.00619 ms` | `0.01072 ms` | `1.73x` |
| B1 / S256 | `0.01127 ms` | `0.01384 ms` | `1.22x` |
| B1 / S384 | `0.01642 ms` | `0.01853 ms` | `1.13x` |
| B2 / S64 | `0.00618 ms` | `0.01037 ms` | `1.68x` |

That comparison includes the allocation/framework cost of eager SDPA. The new
strict phase-compiler calibration also tests allocation-free fixed-address CUDA
graph replay. On the four manifest cells B1/S256, B1/S384, B1/S512, and B2/S128,
the fastest TGV tile reached `0.902x`, `0.869x`, `0.636x`, and `0.987x` of the
fastest correct graph baseline. The compiler therefore keeps all four on an
external exact fallback and retains the native losses as optimization targets.
See [native SM100 GQA prefill](docs/sm100_gqa_prefill_20260828.md) for the
kernel derivation and [the strict calibration](docs/universal_exact_calibration_20260828.md)
for the stronger baseline result.

## Universal Exact Phase Compiler

StreamAttn no longer treats isolated promotion dictionaries as the final engine
architecture. The first universal exact compiler contract is committed in
[`benchmarks/manifests/universal_exact_v1.yaml`](benchmarks/manifests/universal_exact_v1.yaml).
It freezes 30 valid cells instead of constructing an artificial Cartesian
product:

```text
12 real workload cells
10 architecture and scheduler boundary cells
 8 feature-interaction cells

SM80 + SM90 + SM100
decode + prefill + training
FP16 + BF16
D64 + D128 + D256
contiguous + paged, NHD + HND
```

The manifest publishes trace, stratified, and boundary weights; eligible
baselines; numerical tolerances; and compiler acceptance criteria. The physical
IR in `stream_attention/exact_compiler.py` records ownership, algebra
orientation, tile geometry, split strategy, load/MMA engine, accumulator space,
pipeline, scheduler, cluster, softmax, merge, and epilogue. Runtime `B/M/N`
remain in `AttentionProblem` so one compiled binary can cover many cells.

Guarantees are deliberately distinct:

```text
exact                  full-context exact attention
schedule_exact         exact arithmetic over an explicitly selected schedule
distribution_verified  approximation validated at model-output level
```

Current implementations are registered as candidate families, including the
promoted SM90 decode and SM100 prefill kernels, the SM80 `cp.async` decode
kernel, generic native Triton, and an explicitly non-native external fallback.
The manifest exposes a native feature gap: deterministic-dropout training
currently requires an exact PyTorch math fallback.
This is compiler infrastructure, not a new overall performance claim. Inspect
the frozen surface with:

```bash
python benchmarks/inspect_universal_exact_manifest.py
```

See [Universal Exact Phase Compiler v1](docs/universal_exact_phase_compiler_v1.md)
for the invariants and next compiler stages.

The next compiler layer is also available. GPU profilers write immutable
`BackendEvidence` records containing the requested and resolved backend,
environment fingerprint, correctness result, timing distribution, confidence,
workspace, supported range, and timed-allocation count. The phase compiler then:

```text
requires an explicit outcome for every eligible baseline
prefers the fastest correct allocation-free baseline, then a correct allocating fallback
chooses native only when the fastest correct native candidate beats the baseline
otherwise emits an explicit external fallback and retains the native loss
retains failed, unsupported, slower, and losing measurements
computes regret against the fastest valid native-or-external route
writes SHA-indexed phase_db/sm80.json, sm90.json, and sm100.json
```

The SM80 database now resolves all seven declared cells with complete baseline
telemetry and zero routing regret. None is promoted native: D128 paged decode is
the closest at `0.949x` of FlashInfer, while causal/noncausal prefill and
training remain larger kernel-family gaps. Deterministic dropout is exactly
resolved by PyTorch math SDPA, but its eager autograd path allocates, so the
architecture database correctly remains below compiler-v1 acceptance.

A baseline that was never attempted makes the cell unresolved; it cannot be
silently omitted. Current deterministic-dropout training gaps are emitted as
`external_fallback`, while an unmeasured native family is emitted as
`native_unmeasured`. Compile strict evidence with:

```bash
python benchmarks/compile_universal_exact_phase_db.py \
  artifacts/exact/evidence.json \
  --output-dir phase_db
```

See [Exact Phase Database v1](docs/universal_exact_phase_database_v1.md) for the
evidence schema and acceptance semantics. The first partial calibration is now
compiled: four SM90 cells resolve to three native routes and one fallback; four
SM100 cells resolve to external fallbacks against graph-captured baselines. The
remaining cells stay visibly unresolved, so the repository still does not claim
the 30-cell performance target. See [the H100/B200 calibration report](docs/universal_exact_calibration_20260828.md).

## Decode Request Lifecycle

The native engine deliberately separates model-specific work from attention:

```text
request
  -> model adapter projects Q/K/V and applies RoPE
  -> adapter appends K/V to its cache
  -> StreamAttn validates or reuses a fixed-buffer plan
       -> exact_native: stream every KV tile
       -> seed_only_native: stream the validated seed blocks
       -> verified_auto: choose seed only on a matching policy cell
  -> online softmax produces [B, 1, Hq, D]
  -> model adapter applies output projection
  -> sampler chooses the next token
```

Planning is kept outside the steady-state loop. The hot path reuses workspaces,
seed schedules, output buffers, and native cache views instead of rebuilding
Python metadata on every token.

## Online Softmax

For each query row, StreamAttn carries a running maximum `m`, normalizer `l`,
and weighted value numerator `n`. When a new score tile arrives:

```text
m_new = max(m, max(scores_tile))
alpha = exp(m - m_new)
p     = exp(scores_tile - m_new)

l = alpha * l + sum(p)
n = alpha * n + p @ V_tile
m = m_new

output = n / l
```

This is numerically stable, requires linear auxiliary memory, and composes
across split tiles. Exact mode applies it to every KV tile. Seed-only mode uses
the same normalization over the policy-selected KV set.

## Policy Boundaries

Packaged policy cells live in
[`stream_attention/policies/registry.json`](stream_attention/policies/registry.json).
Each cell records the model and layer, tensor space, dtype, KV bucket, minimum
batch, head geometry, seed schedule, safety gates, and measured performance.

A layer that passes in isolation is not automatically safe in a multi-layer
route. Route bundles are evaluated with teacher-forced replay, greedy rollout,
coupled sampling, and adversarial stress prompts. Failed or unknown cells are
not silently promoted.

The dynamic-selector research follows the same rule: query-aware block
selection is interesting only if its selection overhead, model-level safety,
and net decode latency all pass. Research utilities are not part of the default
serving path.

## Benchmarking

The lightweight package CLIs exercise the general fused attention module:

```bash
stream-attention-benchmark \
  --seq 512 1024 2048 4096 \
  --batch 1 --heads 8 --dim 64 --warmup 10 --iters 50

stream-attention-test \
  --seq 1024 --batch 2 --heads 8 --dim 64 --dtype fp16
```

For native decode research, start with the profiler help instead of assuming a
default shape:

```bash
python benchmarks/profile_transposed_wgmma_exact_qk.py --help
python benchmarks/profile_paged_exact_decode.py --help
python benchmarks/profile_seed_only_route_bundle_decode.py --help
python benchmarks/profile_seed_kernel_mode_autotune.py --help
python benchmarks/inspect_universal_exact_manifest.py
python benchmarks/compile_universal_exact_phase_db.py --help
```

A publishable performance result should record:

- GPU model, CUDA, driver, PyTorch, Triton, and baseline versions
- batch, KV length, Q/KV heads, head dimension, dtype, and cache layout
- warmup, repetitions, timing method, and clock/power conditions
- numerical tolerance or model-distribution gate
- paired raw timings, not speedup alone

FlashInfer is the exact decode reference used by the promoted H100 phase
diagrams. PyTorch SDPA is the general correctness fallback and training
reference. FlashAttention-class implementations remain external references,
not hidden StreamAttn dependencies.

## Supported Surface

| Area | Current support |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.1+ |
| CPU | Correctness and SDPA fallback |
| Native GPU evidence | NVIDIA H100 / SM90 and B200 / SM100; promoted A100 / SM80 contiguous D64 cells, one cross-provider contiguous D128 cell, plus complete paged-D128 routing evidence |
| Exact decode | Contiguous BF16 KV; guarded D64/D128 GQA cells plus generic exact fallback |
| Paged exact decode | Direct NHD/HND exact fallback; promoted H100 shape cells plus six B200 NHD/page-16 D128/G8 full-row cells |
| Reduced-work decode | Packaged Qwen-family 32K cells; request-tier and route-bundle restrictions apply |
| Prefill/forward/backward | Public exact MHA/GQA `prefill(...)` and `train(...)` plans; promoted B200 BF16 `Hq16/Hkv2/D128` causal prefill cells; compact Triton online-softmax path with masks, dropout, ALiBi, and autograd; exact SDPA fallback |
| Distributed research | Ring and Star attention prototypes |
| Experimental hardware | A100 has promoted contiguous D64 and B4/G8/16K D128 `cp.async` + MMA exact decode; paged D128 still falls back externally |
| Not yet promoted | H100 WGMMA prefill, remaining A100 D128 shapes plus prefill/backward, other B200 prefill/decode shapes, ragged page-64 WGMMA, FP8 selected cache, second model family |

## Repository Guide

```text
stream_attention/
  engine.py                 public decode, prefill, and training engine
  functional.py             planned exact prefill/training execution
  exact_compiler.py         workload, schedule, and resource compiler IR
  phase_database.py         strict evidence resolution and phase-table compiler
  decode.py                 native modes, planning, and fail-closed service
  backends/sm80/            promoted contiguous D64/D128 and experimental paged-D128 Ampere exact kernels
  backends/sm90/            promoted Hopper exact kernels and dispatch
  backends/sm100/           promoted Blackwell exact backend and headers
  kernels/                  Triton/CUDA attention kernels
  policies/                 calibrated policy cells and route bundles
  core/                     general forward/backward attention modules
  integration/              model integration helpers

benchmarks/                 profilers, manifests, compilers, and evidence tools
docs/                       phase diagrams and research decisions
examples/                   minimal usage and integration examples
tests/                      CPU, policy, API, and CUDA-gated tests
```

The detailed experiment history is intentionally kept in `docs/` rather than
the README. Start with:

- [Documentation index](docs/Index.md)
- [Qwen2.5-3B route evidence](docs/qwen25_3b_32k_b4_seed_policy.md)
- [Dynamic selector findings](docs/qwen25_3b_dynamic_selector_findings.md)
- [D128 pipeline ablation](docs/sm90_d128_pipeline_ablation_20260819.md)
- [Paged exact decode](docs/paged_exact_decode.md)
- [Universal Exact Phase Compiler v1](docs/universal_exact_phase_compiler_v1.md)
- [Exact Phase Database v1](docs/universal_exact_phase_database_v1.md)
- [Literature and backend decision ledger](docs/streamattn_literature_decision_ledger.md)

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. The
required CI result covers policy integrity, Python 3.10/3.11 CPU tests, package
validation, and GPU-source contracts. CUDA-source changes also trigger an
offline SM90 compile check on a standard hosted runner; contributors are not
expected to pay for GPU CI.

Actual-device performance claims still require a reproducible hardware
artifact. The contributor harness and evidence requirements are documented in
[GPU CI and wheels](docs/gpu_ci_and_wheels.md).

Useful local checks:

```bash
python -m pytest -q
python benchmarks/check_gpu_source_contract.py
python -m build
python benchmarks/check_wheel_contents.py dist/*.whl
```

## Roadmap

The project has proved that StreamAttn-owned exact kernels can beat strong
baselines on guarded Hopper and Blackwell cells. The next objective is broad,
reproducible exact coverage rather than another isolated promotion:

1. Complete the remaining SM90 and SM100 manifest cells with the same strict
   evidence contract now used by the complete seven-cell SM80 database.
2. Connect compiled `ptxas`/occupancy reports to the SM80/SM90/SM100 resource
   legality models, then add analytical roofline pruning and active exploration
   near uncertain phase boundaries.
3. Expand B200 prefill through the compiler across G4/G8/G16, D64/D128/D256,
   FP16/BF16, causal/noncausal, and the short-M to long-M transition.
4. Independently replay the 11 winning cells from the completed 30-cell H100
   consumer-owned RS-PV prefill screen, then insert only replicated winners into
   the exact phase database and guarded public dispatch. Expand that family
   across query-head counts, GQA groups, D64/D256, FP16, noncausal attention,
   and ragged lengths; keep every unmeasured or losing cell on exact fallback.
5. Split native backward into query-owned dQ and KV-group-owned dK/dV families
   with deterministic partial reductions instead of global GQA atomics.
6. Extend the register-resident A100 D128 producer beyond the promoted
   B4/G8/16K cell. Strict 32K, G4, and B8 tests rejected the current schedule,
   so the next step is a different work decomposition rather than more buffer
   or merge-split permutations; then build distinct prefill and backward
   schedules.
7. Resume adaptive/selected work above the exact compiler. Unknown or failed
   guarantees must return to StreamAttn exact execution.

The public engine API is now:

```python
stream_attn.decode(...)
stream_attn.prefill(...)
stream_attn.train(...)
```

with native exact, seed-only, adaptive, and verified routes behind one engine.
Decode has the broadest optimized backend portfolio today. Prefill and training
now execute exact native MHA and compact GQA, while the grouped-KV
architecture-specific performance backends are built.

## License

Apache-2.0. See [LICENSE](LICENSE).
