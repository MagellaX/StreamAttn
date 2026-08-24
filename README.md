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

> **Project status:** research engine with guarded H100 routes. StreamAttn has
> apples-to-apples exact decode wins over FlashInfer on promoted shapes and a
> measured model-level Qwen decode win on a validated request tier. It is not a
> universal replacement for FlashInfer, FlashAttention, or a full serving
> runtime yet.

## Why StreamAttn Exists

Fast exact kernels answer this question:

```text
How efficiently can the GPU compute all requested attention work?
```

StreamAttn also asks:

```text
What is the cheapest native attention route that is valid for this request?
```

That produces three decode modes:

| Mode | Work performed | Semantics | Current use |
|---|---|---|---|
| `exact_native` | All KV tokens | Exact attention, within numerical tolerance | Default and fail-closed route |
| `seed_only_native` | A small sink/middle/recent seed set | Approximate; only for a packaged, validated policy cell | Explicit opt-in and calibrated serving |
| `verified_auto` | Seed-only when policy invariants match, otherwise exact | Policy-verified, fail-closed routing | Default planning mode; live generic verification is still research |

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
exact schedules. Selected paged work now compiles into a device CSR schedule
and page-native `PackedRoute64` metadata without copying K/V. The H100
selected-paged WGMMA executor is the remaining promotion boundary; no speedup
is claimed for that path yet. The shared planner means future adaptive,
compressed, prefill, and device backends do not need a second semantic API or
a second online-softmax model. See [the universal tile
planner](docs/universal_attention_tile_planner.md) and [the selected paged
route ABI](docs/selected_paged_route_abi.md).

## What Is Proven Today

### Exact native decode

The promoted SM90 kernels compute full-context GQA decode and use online
softmax plus exact split-state merging. Contiguous and page-64 results below
use FlashInfer 0.6.12; page-16 was re-gated against FlashInfer 0.6.17 on H100:

| Shape family | Measured region | StreamAttn speedup | Status |
|---|---|---:|---|
| D64, GQA group 8 | 7 cells; B2-B8; 16K-64K KV | `1.025x-1.432x` | Promoted per cell |
| D64, GQA group 4 | 14 cells; B1-B16; 16K-64K KV | `1.027x-1.449x` | Promoted per cell |
| D128, GQA group 4 | 6 cells; B4-B16; 16K-64K KV | `1.002x-1.012x` | Promoted per cell |
| Paged D64, GQA group 8 | HND/page-64; B1-B8; 16K-64K KV | `1.21x-2.24x` paired median | Promoted per cell |
| Paged D64, GQA group 8 | HND/page-16 full rows; B1-B8; 16K-64K capacity | `1.21x-2.07x` paired median | Promoted per cell |
| Paged D64, GQA group 8 | HND/page-16 ragged rows; same capacity matrix | `1.17x` worst paired trial; `2.04x` median cell | Promoted per cell |
| Paged D128, GQA group 8 | HND/page-16; B1-B8; 16K-64K; full and ragged | `1.075x` worst paired trial; `1.75x` median auto-gate cell | Promoted per cell |
| Paged D128, GQA group 8 | NHD/page-16; B1-B8; 16K-64K; full and ragged | `1.058x` worst paired trial; `1.283x` median auto-gate cell | Promoted per cell |
| Paged D128, GQA group 4 | HND/page-16; B1-B8; 16K-64K; full and ragged | `1.017x` worst paired trial; `1.34x` median cell | Promoted per cell |
| Paged D128, GQA group 8 | A100/NHD/page-16; B1/32K full row | `1.20x-1.26x` in two repeated runs; `0.975x` in one warm-state sweep | Experimental candidate; not auto-routed |
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
- [Native B200 exact backend and promotion boundary](docs/paged_exact_sm100_tgv_20260824.md)

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
- No A100, FP8, or non-Qwen model-family performance claim has been promoted
  yet. A100 now has an architecture-native SM80 `cp.async` + BF16 MMA
  exact backend for direct NHD page-16 D128/G8 decode. It is correct and has
  repeated B1/32K wins, but another independent warm-state sweep reached only
  `0.975x`; it therefore remains explicit experimental opt-in.
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

### General forward and training module

`StreamAttention` provides the fused online-softmax forward/backward path for
standard transformer experimentation:

```python
import torch
from stream_attention import StreamAttention, StreamAttentionConfig

config = StreamAttentionConfig(num_heads=8, head_dim=64)
attention = StreamAttention(config).cuda()

q = torch.randn(2, 1024, 8, 64, device="cuda", dtype=torch.float16,
                requires_grad=True)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = attention(q, k, v, causal=True)
output.square().mean().backward()
```

The Triton path supports boolean/additive masks, dropout, deterministic Philox
seeding, ALiBi, and a streaming backward pass. Unsupported environments fall
back to PyTorch SDPA.

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
| Native GPU evidence | NVIDIA H100 / SM90 and B200 / SM100; A100 / SM80 experimental |
| Exact decode | Contiguous BF16 KV; guarded D64/D128 GQA cells plus generic exact fallback |
| Paged exact decode | Direct NHD/HND exact fallback; promoted H100 shape cells plus six B200 NHD/page-16 D128/G8 full-row cells |
| Reduced-work decode | Packaged Qwen-family 32K cells; request-tier and route-bundle restrictions apply |
| Forward/backward | Triton online-softmax path with masks, dropout, ALiBi, and autograd |
| Distributed research | Ring and Star attention prototypes |
| Experimental hardware | A100 has native `cp.async` + MMA exact decode with a variable B1/32K candidate edge |
| Not yet promoted | A100, other B200 shapes/ragged rows, ragged page-64 WGMMA, FP8 seed cache, second model family |

## Repository Guide

```text
stream_attention/
  engine.py                 public fixed-buffer decode engine
  decode.py                 native modes, planning, and fail-closed service
  backends/sm80/            experimental Ampere exact kernels
  backends/sm90/            promoted Hopper exact kernels and dispatch
  backends/sm100/           promoted Blackwell exact backend and headers
  kernels/                  Triton/CUDA attention kernels
  policies/                 calibrated policy cells and route bundles
  core/                     general forward/backward attention modules
  integration/              model integration helpers

benchmarks/                 profilers, policy compilers, and evidence tools
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

The project has completed the first proof: StreamAttn-owned exact kernels can
beat a strong exact decode baseline on guarded H100 cells, and a calibrated
reduced-work route can speed up complete model decode. The next milestones are:

1. Execute the new device CSR -> `PackedRoute64` selected-paged ABI in the
   H100 transposed WGMMA backend, then gate static versus compact-ragged
   scheduling with measured route variance.
2. Replace offline verification schedules with a selective live runtime
   verifier.
3. Add a second model family to test whether policy discovery generalizes
   beyond Qwen.
4. Add first-class `prefill(...)` and `train(...)` entry points through the
   same `AttentionProblem -> AttentionTilePlan -> AttentionBackendPlan`
   contract.
5. Lower the selected-route ABI into the B200 TMA+TMEM+`tcgen05` backend and
   expand it beyond the promoted
   page-16 NHD D128/G8 full-row cells, and finish the A100 PV
   shared-to-register transpose so the SM80 candidate scales beyond B1.
6. Improve B1/B2 economics with a single-kernel cooperative selected path.
7. Promote query-aware dynamic selection only where it beats exact fallback
   after selector overhead.
8. Evaluate FP8/FP4 selected-cache paths under the same distribution-level
   gates.

The long-term API target is:

```python
stream_attn.decode(...)
stream_attn.prefill(...)
stream_attn.train(...)
```

with native exact, seed-only, adaptive, and verified routes behind one engine.

## License

Apache-2.0. See [LICENSE](LICENSE).
