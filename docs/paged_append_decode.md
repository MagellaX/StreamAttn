# Owner-Only Append And Decode

`PagedAppendDecodePlan` is an opt-in full-attention path for H100: the producer
CTA that reads a new KV token also writes it into its exclusively owned cache
tail. This removes the separate append launch. It does not omit attention work,
change normalized partials, or replace the existing exact merge.

Ordinary `PagedExactDecodePlan` dispatch is unchanged. This API is for callers
that have incoming post-RoPE K/V and still need to append it. It is not a claim
of savings when an upstream QKV/RoPE kernel has already written the cache.

## Supported Contract

- H100/SM90, BF16, one decode query: `query[1,1,16,128]`.
- Two KV heads, G8, contiguous NHD pages: `[num_pages,16,2,128]`.
- Int32 page table with 2,048 entries; 32,768-token capacity.
- Incoming `new_key` and `new_value`: `[1,2,128]`.
- Full visible support, positive length including the incoming token.
- Split count 64 or 128. The default 128 is the retained campaign winner.
- Stable buffer allocations and exclusive writable-tail ownership.

```python
from stream_attention import PagedAppendDecodePlan

# cache is a PagedKVCache. sequence_lengths includes the new logical token.
plan = PagedAppendDecodePlan.build(
    query, cache, new_key, new_value,
    exclusive_tail_ownership=True,
)
output = plan.run()
```

Update the contents of the bound buffers before subsequent calls; do not resize
or replace their storage. The caller must publish valid page mappings and
lengths before execution. This API does not allocate pages or increment lengths.
It appends at `sequence_lengths[0] - 1`, then computes full attention.

## Ownership Is Required

For each KV head, one CTA owns the 64-token tile containing the appended token.
It writes that head's incoming K/V and synchronizes its threads before loading
the tile. Other CTAs must not read those writable elements. The barrier is not
a publication mechanism for another CTA or a concurrent stream.

For visible length `N` and `S` splits, there are `ceil(N/64)` logical tiles,
with `ceil(ceil(N/64)/S)` tiles per split. The tail tile `(N-1)//64` belongs to
exactly one nonempty split. This remains true as length changes and ownership
moves to another split; no historical owner ID is cached.

Setup checks shapes, dtypes, devices, current active pages, buffer overlaps,
and whether the writable tail page aliases an earlier active logical page.
Immutable prefix pages may be shared; writable tail pages may not.

`exclusive_tail_ownership=True` is the allocator/caller's explicit guarantee
about other requests and streams. Inspecting one request cannot discover
unrelated readers. The caller must maintain this guarantee across metadata
changes and serialize overlapping users of cache, workspace and output.

Build-time preflight synchronizes and is not free recurring work. `run()` does
not read device metadata back to Python or revalidate ownership. `validate()`
provides a synchronizing check of the current snapshot when needed. If an
allocator cannot maintain ownership, its repeated validation or repair cost
belongs in complete-call latency. CUDA graph replay cannot execute Python
preflight; raw device mutations do not become safe merely because setup passed.

## Measured Result

The retained owner-only candidate was measured on twelve saved H100 states:
two Qwen2.5-Coder-3B requests, three layers, and decode steps 1 and 16 near 32K.
Its comparison includes append, required metadata updates, attention and merge.

| Cache condition | Speedup vs better tested production control | Speedup vs fastest tested FlashInfer |
| --- | --- | --- |
| Warm | 1.0861x | 1.1389x |
| Four-cache rotation | 1.0830x | 1.0641x |

The production controls used both 64 and 128 splits. FlashInfer 0.6.13 FA2 was
fastest among six validated routes. Two confirmations gave 1,008 positive
paired comparisons; these remain twelve states, not 1,008 independent workloads.
Same-geometry outputs were byte-identical. Setup/allocation and upstream
QKV/RoPE generation were excluded; eager host dispatch was measured separately.
Four-cache rotation is not a claim of cold-cache execution.

The numbers above describe the retained measured candidate, not a new whole-model
or cross-architecture result. Raw-state export was slower and is not integrated.
The source transfer preserves the original producer ABI and uses an isolated
owner-only specialization with the same normalized producer arithmetic and merge.

The integrated API passed a targeted H100 transfer check: all twelve saved
states, sixteen consecutive captured decode steps, page/K64 and split-ownership
transitions at both split counts, and a derived two-plan fixture sharing only
immutable prefix pages. All 64 main output checkpoints matched the retained native
computation bit for bit, with cache checks. Memcheck, racecheck and synccheck
also passed their targeted boundary/shared-prefix fixtures.

A one-state integration parity check, not a repeated FlashInfer campaign,
measured 16.381 us warm and 24.772 us with rotation, versus the retained
candidate's 16.362 us and 24.641 us. Eager medians were 18.750 us integrated
and 18.787 us retained. The transfer stayed within 0.6% of the retained graph
medians; it is not claimed as an additional speedup. Changing-state correctness
tests do not establish whole-model latency or timed allocator/page-creation cost.

Native numerical agreement remains distinct from the separate `0.001` omission
allowance. This path introduces no omission or new approximation; byte identity
against same-geometry native output is not `0.001` absolute agreement to FP64.

Future adaptive comparisons must use owner-fused full attention wherever that
ownership is valid, and include live decisions and upkeep. Append savings must
not be counted as an adaptive advantage over an artificially unfused control.

## Evidence Identity

The retained campaign was based on source commit
`aa6452953568b91310ab59a834cb22f3a02870b2`.
Its candidate binary SHA256 is
`6e459614ff25441accf336295f96f662388c5e6c0fc8b019e713508966b33638`.
The independent replay archive SHA256 is
`65dccda821b7e4bdd904ccd66202ca4fa6714cc808cb8bb69838fe1a9f28188c`.
Raw timings, reference outputs, cache checks, counter captures and the rejected
raw-state arms remain archived in the external workbench, not bundled into the
package or represented as a default adaptive route.

Integrated D128 binary SHA256:
`93d13cdd6d7cd6b769f15223c8b4d96f0ad349a07250d8f23d0e7902a81af848`.
Targeted H100 transfer archive SHA256:
`c3e0a82a403a2108b853d1c95a235d8e6efa4732bd61162172d612f6b0a644c4`.
