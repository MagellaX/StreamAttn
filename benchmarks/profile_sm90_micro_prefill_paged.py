"""Direct paged/ragged correctness, poisoned tails and mutable graph replay.

Reference gathering is outside the native path and all timed regions. This
experiment establishes semantics, not promotion against an external backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profile_sm90_micro_prefill import _capture, _elapsed_graph_ms
from benchmarks.profile_sm90_micro_prefill_semantics import check_output, fp32_reference
from stream_attention.paged import PagedKVCache
from stream_attention.backends.sm90.micro_prefill_paged import PagedMicroPrefillPlan, validate_paged_micro_prefill

SCHEMA = "streamattn.sm90_micro_prefill_paged.v1"
SOURCE_PATHS = (
    "stream_attention/paged.py",
    "stream_attention/backends/sm90/micro_prefill_paged.py",
    "stream_attention/backends/sm90/micro_prefill_paged_sources.py",
    "stream_attention/backends/sm90/micro_prefill_semantics.py",
    "stream_attention/backends/sm90/micro_prefill_semantics_sources.py",
    "stream_attention/backends/sm90/transposed_gqa_exact_sources.py",
    "benchmarks/profile_sm90_micro_prefill_paged.py",
    "benchmarks/profile_sm90_micro_prefill_semantics.py",
)


def experiment_cases(suite):
    geometries = [
        (4, 3, 5, 16, 4, 2),
        (2, 17, 29, 16, 8, 7),
        (1, 2, 1, 16, 8, 1),
        (2, 64, 256, 16, 8, 16),
        (4, 9, 1025, 32, 4, 32),
        (1, 32, 3, 32, 4, 1),
    ]
    if suite == "smoke":
        return [dict(batch=4, m=3, pages=5, hq=16, g=4, splits=2,
                     d=128, dtype="bf16", mask="permuted", layout=layout)
                for layout in ("HND", "NHD")]
    if suite == "replay":
        geometries = geometries[:2]
    return [dict(batch=b, m=m, pages=p, hq=h, g=g, splits=s,
                 d=d, dtype=dtype, mask=mask, layout=layout)
            for d in (64, 128) for dtype in ("bf16", "fp16")
            for b, m, p, h, g, s in geometries
            for layout in ("HND", "NHD")
            for mask in ("append", "permuted", "noncausal")]


def set_metadata(c, q, cache, ql, qp, kp, *, mutate=False, empty=False):
    b, m, n = c["batch"], c["m"], cache.max_sequence_length
    if empty:
        cache.sequence_lengths.zero_()
        cache.page_table.fill_(-1)
        ql.fill_(m)
    else:
        lengths = [n - 1] if b == 1 else [min(n, x) for x in ([1, n - 3] if b == 2 else [0, 1, n - 17, n])]
        queries = [m] if b == 1 else ([m, m - 1] if b == 2 else [0, 1, m - 1, m])
        if mutate:
            lengths = [max(0, n - x) for x in lengths]
            queries = [min(m, max(0, m - x + 1)) for x in queries]
        cache.sequence_lengths.copy_(torch.tensor(lengths, dtype=torch.int32, device=q.device))
        ql.copy_(torch.tensor(queries, dtype=torch.int32, device=q.device))
        table = torch.randperm(cache.num_pages, device=q.device)[:b*c["pages"]].reshape(b, c["pages"])
        if b > 1 and lengths[-1] >= 16 and lengths[-2] >= 16:
            table[-1, 0] = table[-2, 0]  # shared prefix; never assume exclusive physical pages
        active = torch.arange(c["pages"], device=q.device)[None, :] * 16 < cache.sequence_lengths[:, None]
        cache.page_table.copy_(table.masked_fill(~active, -1))
    q.normal_().mul_(3 if mutate else 1)
    cache.key.normal_()
    cache.value.normal_()
    # Mark valid physical tokens as a union so shared-prefix pages stay valid.
    valid = torch.zeros(cache.num_pages, 16, dtype=torch.bool, device=q.device)
    for row, length in enumerate(cache.sequence_lengths.cpu().tolist()):
        if length:
            token = torch.arange(length, device=q.device)
            valid[cache.page_table[row, token // 16].long(), token % 16] = True
    mask = valid[:, :, None, None] if cache.normalized_layout == "NHD" else valid[:, None, :, None]
    cache.key.masked_fill_(~mask, float("nan"))
    cache.value.masked_fill_(~mask, float("nan"))
    q.masked_fill_(torch.arange(m, device=q.device)[None, :, None, None] >= ql[:, None, None, None], float("nan"))
    origin = (torch.arange(b, device=q.device, dtype=torch.int64) * 10000 + (1 << 40))[:, None]
    kp.copy_(torch.arange(n, device=q.device)[None, :] + origin)
    qp.copy_(torch.arange(m, device=q.device)[None, :] + origin + cache.sequence_lengths[:, None] - ql[:, None])
    if c["mask"] == "permuted" or mutate:
        kp.copy_(kp.flip(1))
        qp[:, 0] = origin[:, 0] - 1
        qp[:, 1] = origin[:, 0] + n - 1


def reference(q, cache, ql, qp, kp, causal):
    b, m, h, d = q.shape
    output = torch.zeros(b, m, h, d, dtype=torch.float32, device=q.device)
    lse = torch.full((b, m, h), -torch.inf, dtype=torch.float32, device=q.device)
    for row, (nq, nk) in enumerate(zip(ql.cpu().tolist(), cache.sequence_lengths.cpu().tolist())):
        if not nq or not nk:
            continue
        token = torch.arange(nk, device=q.device)
        pages = cache.page_table[row, token // 16].long()
        if cache.normalized_layout == "NHD":
            k = cache.key[pages, token % 16].permute(1, 0, 2)
            v = cache.value[pages, token % 16].permute(1, 0, 2)
        else:
            k = cache.key[pages, :, token % 16].permute(1, 0, 2)
            v = cache.value[pages, :, token % 16].permute(1, 0, 2)
        out, norm = fp32_reference(
            q[row:row+1, :nq], k[None], v[None],
            qp[row:row+1, :nq] if causal else None,
            kp[row:row+1, :nk] if causal else None,
        )
        output[row, :nq], lse[row, :nq] = out[0], norm[0]
    return output, lse


def reconstructed_lse(plan):
    b, m, h, _ = plan.query.shape
    hk, g = plan.cache.kv_heads, h // plan.cache.kv_heads
    merged = torch.logsumexp(plan.partial_lse * math.log(2), 1)
    if not plan.natural:
        return merged.view(b, m, hk, 8)[..., :g].reshape(b, m, h)
    qt = (m + 64 // g - 1) // (64 // g)
    return (merged.view(b, hk, qt, 64 // g, g).reshape(b, hk, qt * (64 // g), g)
            [:, :, :m].permute(0, 2, 1, 3).reshape(b, m, h))


def profile_case(c, args):
    torch.manual_seed(args.seed + c["m"] + c["pages"] + c["d"])
    dtype = torch.float16 if c["dtype"] == "fp16" else torch.bfloat16
    b, m, h, d, p, hk = c["batch"], c["m"], c["hq"], c["d"], c["pages"], c["hq"] // c["g"]
    q = torch.empty(b, m, h, d, device="cuda", dtype=dtype)
    shape = (b*p+3, 16, hk, d) if c["layout"] == "NHD" else (b*p+3, hk, 16, d)
    cache = PagedKVCache(
        torch.empty(shape, device="cuda", dtype=dtype), torch.empty(shape, device="cuda", dtype=dtype),
        torch.empty(b, p, device="cuda", dtype=torch.int32),
        torch.empty(b, device="cuda", dtype=torch.int32), c["layout"],
    )
    ql = torch.empty(b, device="cuda", dtype=torch.int32)
    qp = torch.empty(b, m, device="cuda", dtype=torch.int64)
    kp = torch.empty(b, p*16, device="cuda", dtype=torch.int64)
    set_metadata(c, q, cache, ql, qp, kp)
    causal = c["mask"] != "noncausal"
    kwargs = dict(causal=True, query_positions=qp, key_positions=kp) if causal else {}
    plans = {name: PagedMicroPrefillPlan.build(
        q, cache, ql, natural=natural, num_splits=c["splits"],
        cutlass_root=args.cutlass_root, build_dir=args.build_dir,
        compile_verbose=True, **kwargs,
    ) for name, natural in (("transposed", False), ("natural", True))}
    graphs, rows = {}, {}
    pointers = [t.data_ptr() for t in (q, cache.key, cache.value, cache.page_table, cache.sequence_lengths, ql, qp, kp)]
    for stage in ("eager", "mutated_graph", "empty_graph"):
        if stage != "eager":
            set_metadata(c, q, cache, ql, qp, kp, mutate=True, empty=stage == "empty_graph")
        validate_paged_micro_prefill(q, cache, ql, **kwargs)
        expected, expected_lse = reference(q, cache, ql, qp, kp, causal)
        for name, plan in plans.items():
            plan.output.fill_(float("nan"))
            plan.partial_output.fill_(float("nan"))
            plan.partial_lse.fill_(float("nan"))
            plan.run() if stage == "eager" else graphs[name].replay()
            torch.cuda.synchronize()
            rows.setdefault(name, {})[stage] = check_output(
                plan, expected, expected_lse, observed_lse=reconstructed_lse(plan),
            )
            if not rows[name][stage]["passed"]:
                return dict(case=c, families=rows, passed=False)
            if stage == "eager":
                graphs[name] = _capture(plan.run, warmup=3)
                rows[name]["complete_graph_us"] = [
                    1000 * _elapsed_graph_ms(graphs[name], iterations=30) for _ in range(3)
                ]
                rows[name]["workspace_bytes"] = plan.workspace_bytes
    assert pointers == [t.data_ptr() for t in (q, cache.key, cache.value, cache.page_table, cache.sequence_lengths, ql, qp, kp)]
    return dict(case=c, families=rows, passed=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("smoke", "replay", "full"), default="full")
    parser.add_argument("--provider", default="local")
    parser.add_argument("--seed", type=int, default=7013)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError("preserve existing evidence")
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = experiment_cases(args.suite)
    result = dict(schema=SCHEMA, complete=False, provider=args.provider, seed=args.seed,
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  device=torch.cuda.get_device_name(), planned_cases=len(cases), rows=[],
                  source_sha256={p: hashlib.sha256((ROOT / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                 for p in SOURCE_PATHS})
    try:
        for c in cases:
            print(json.dumps(dict(stage="case", case=c)), flush=True)
            row = profile_case(c, args)
            result["rows"].append(row)
            print(json.dumps(dict(stage="checked", case=c, passed=row["passed"])), flush=True)
            if not row["passed"]:
                raise AssertionError(f"paged correctness failed: {c}")
        result["passed_cases"] = len(result["rows"])
        result["complete"] = True
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
