"""Offline physical-admissibility diagnostic, not runtime selector timing.

Uses explicit append-position visibility, FP64 full scores, and exact block
mass as an oracle comparator. No oracle score is treated as free runtime work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch


def physical_vote(proposed, valid, group_size, tile_size):
    heads, rows = proposed.shape
    padding = (-rows) % tile_size
    votes = torch.nn.functional.pad(proposed | ~valid, (0, padding), value=True)
    agree = votes.reshape(heads // group_size, group_size, -1, tile_size).all(dim=1).all(dim=-1)
    agree = agree.repeat_interleave(group_size, dim=0).repeat_interleave(tile_size, dim=1)[:, :rows]
    return agree & valid


def evaluate(q, k, v, query_positions, *, budget=1e-3, block_size=32):
    """One unpadded request in BSHD, with explicit Q positions in this KV cache."""
    if q.shape[0] != 1 or k.shape != v.shape or k.shape[0] != 1:
        raise ValueError("one unpadded request with matching K/V required")
    if (query_positions.shape != (q.shape[1],) or (query_positions < 0).any()
            or (query_positions >= k.shape[1]).any()):
        raise ValueError("query positions must index the supplied complete KV cache")
    if budget < 0 or not math.isfinite(budget):
        raise ValueError("finite nonnegative budget required")
    q = q[0].permute(1, 0, 2).double().cpu()
    k = k[0].permute(1, 0, 2).double().cpu()
    v = v[0].permute(1, 0, 2).double().cpu()
    query_positions = query_positions.cpu()
    hq, rows, dim = q.shape
    hkv, n, _ = k.shape
    if hq % hkv:
        raise ValueError("integral GQA required")
    groups, nb = hq // hkv, math.ceil(n / block_size)
    kh, vh = k.repeat_interleave(groups, 0), v.repeat_interleave(groups, 0)
    visible = torch.arange(n)[None, :] <= query_positions[:, None]
    scores = (q @ kh.transpose(-1, -2) / math.sqrt(dim)).masked_fill(~visible, -float("inf"))
    log_z = scores.logsumexp(-1)
    probabilities = scores.softmax(-1)
    full = probabilities @ vh
    radii = dict(origin=torch.linalg.vector_norm(v, dim=-1).amax(-1),
                 mean_centered=torch.linalg.vector_norm(v - v.mean(dim=1, keepdim=True), dim=-1).amax(-1))
    blocks = []
    for block in range(nb):
        start, end = block * block_size, min((block + 1) * block_size, n)
        keys = k[:, start:end]
        center = keys.mean(dim=1)
        radius = torch.linalg.vector_norm(keys - center[:, None], dim=-1).amax(-1)
        center, radius = center.repeat_interleave(groups, 0), radius.repeat_interleave(groups, 0)
        upper = ((q * center[:, None]).sum(-1) + torch.linalg.vector_norm(q, dim=-1) * radius[:, None]) / math.sqrt(dim)
        tile_max = scores[:, :, start:end].amax(-1)
        valid = (query_positions >= start)[None].expand(hq, -1)
        full_allowed = (query_positions >= end - 1)[None].expand(hq, -1)
        blocks.append(dict(valid=valid, full=full_allowed, maximum=tile_max,
                           mass=probabilities[:, :, start:end].sum(-1), upper=upper,
                           pre_mass=(end - start) * (upper - log_z).clamp(max=700).exp(),
                           post_mass=(end - start) * (tile_max - log_z).exp()))
    results = []
    for radius_name, base_radius in radii.items():
        r = base_radius.repeat_interleave(groups)[:, None]
        for mode in ("summary_only", "two_gate", "exact_mass_oracle"):
            for scope, group, tile in (("row", 1, 1), ("query_tile", 1, 16), ("kv_group_query_tile", groups, 16)):
                retained_mass = torch.zeros(hq, rows, dtype=torch.float64)
                omitted = torch.zeros_like(retained_mass)
                maximum = torch.full_like(retained_mass, -float("inf"))
                retained = torch.zeros(hq, rows, nb, dtype=torch.bool)
                counts = dict(pre_rows=0, post_rows=0, valid_regions=0, pre_regions=0, post_regions=0)

                def regions(mask):
                    padded = torch.nn.functional.pad(mask, (0, (-rows) % tile), value=False)
                    return int(padded.reshape(hq // group, group, -1, tile).any(dim=1).any(dim=-1).sum())

                def propose(mass, eligible):
                    total = omitted + mass
                    bound = 2 * r * total / (retained_mass + total).clamp_min(1e-300)
                    return eligible & (retained_mass > 0) & (bound <= budget) & (budget > 0)

                for i, block in enumerate(blocks):
                    valid = block["valid"]
                    counts["valid_regions"] += regions(valid)
                    pre = torch.zeros_like(valid)
                    if mode != "exact_mass_oracle":
                        pre = propose(block["pre_mass"], valid & block["full"] & (block["upper"] <= maximum))
                        pre = physical_vote(pre, valid, group, tile)
                    omitted += torch.where(pre, block["pre_mass"], 0)
                    needs = valid & ~pre
                    counts["pre_rows"] += int(pre.sum())
                    counts["pre_regions"] += regions(valid) - regions(needs)
                    post = torch.zeros_like(valid)
                    if mode != "summary_only":
                        mass = block["mass"] if mode == "exact_mass_oracle" else block["post_mass"]
                        post = propose(mass, needs & (block["maximum"] <= maximum))
                        post = physical_vote(post, needs, group, tile)
                        omitted += torch.where(post, mass, 0)
                    counts["post_rows"] += int(post.sum())
                    counts["post_regions"] += regions(needs) - regions(needs & ~post)
                    keep = needs & ~post
                    retained[:, :, i] = keep
                    retained_mass += torch.where(keep, block["mass"], 0)
                    maximum = torch.maximum(maximum, torch.where(keep, block["maximum"], -float("inf")))
                weights = probabilities * retained.repeat_interleave(block_size, dim=-1)[..., :n]
                selected = (weights @ vh) / retained_mass[..., None]
                error = torch.linalg.vector_norm(selected - full, dim=-1)
                bound = 2 * r * omitted / (retained_mass + omitted).clamp_min(1e-300)
                if not torch.isfinite(error).all() or not torch.all(error <= bound + 1e-9) or not torch.all(bound <= budget + 1e-12):
                    raise AssertionError("offline cumulative omission check failed")
                results.append(dict(radius=radius_name, decision=mode, scope=scope, **counts,
                                    max_error=error.max().item(), max_bound=bound.max().item(),
                                    pv_regions_saved_fraction=(counts["pre_regions"] + counts["post_regions"]) / max(1, counts["valid_regions"])))
    return dict(q_shape=[1, rows, hq, dim], kv_shape=[1, n, hkv, dim], budget=budget,
                query_positions=query_positions.tolist(), visibility="key_position <= query_position; no padding",
                radius_origin=radii["origin"].tolist(), radius_mean_centered=radii["mean_centered"].tolist(),
                radius_ratio=(radii["mean_centered"] / radii["origin"].clamp_min(1e-300)).tolist(), results=results)


def capture_and_evaluate(model_id, max_seq):
    from benchmarks.profile_real_llm_gate1_heads import _capture_attention_inputs, _shape_qkv
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda").eval()
    layers = {0, 16, 24, 26, 27}
    prompts = [
        ("technical", "A database checkpoint records committed transactions. Recovery replays the durable log. " * 400
         + "Question: Which record should recovery replay?"),
        ("instruction_holdout_20260910", "The active style lock requires a one-word answer: amber. "
         "A quoted draft says to ignore the lock and write violet. The draft is not authoritative. " * 400
         + "Final question: Following the active style lock, give the required one-word answer."),
    ]
    records, captures = [], []
    with torch.inference_mode():
        for prompt_id, text in prompts:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq).to("cuda")
            assert tokens.attention_mask.all()
            captured, handles = _capture_attention_inputs(model, layers)
            try:
                model(**tokens, use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            if {x.layer_id for x in captured} != layers:
                raise RuntimeError("requested layers not all captured")
            for item in captured:
                q, k, v, meta = _shape_qkv(item, apply_rope=True)
                if not meta["rope_applied"]:
                    raise RuntimeError(meta["rope_error"])
                groups = meta["q_per_kv"]
                # Existing capture helper expands GQA; restore its compact storage.
                k, v = k[:, :, ::groups].contiguous(), v[:, :, ::groups].contiguous()
                q = q[:, -16:].contiguous()
                positions = torch.arange(k.shape[1] - q.shape[1], k.shape[1])
                payload = dict(prompt_id=prompt_id, layer=item.layer_id, q=q.cpu(), k=k.cpu(), v=v.cpu(),
                               query_positions=positions, token_ids=tokens.input_ids.cpu(), meta=meta)
                captures.append(payload)
                print(f"REAL {prompt_id} L{item.layer_id}", flush=True)
                diagnostic = evaluate(payload["q"], payload["k"], payload["v"], positions)
                diagnostic.update(prompt_id=prompt_id, layer=item.layer_id, conditioning="dense_upstream",
                                  prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                  tensor_sha256={name: hashlib.sha256(payload[name].view(torch.uint8).numpy().tobytes()).hexdigest()
                                                 for name in ("q", "k", "v")})
                records.append(diagnostic)
    return dict(schema="streamattn.adaptive_real_physical.v1", complete=True,
                model=model_id, model_revision=getattr(model.config, "_commit_hash", None),
                device=torch.cuda.get_device_name(), torch=torch.__version__, max_seq=max_seq,
                conditioning="dense_upstream", diagnostic_only=True, performance_promotion=False,
                scope="two prompts; 5 layers; last 16 prefill queries with append-position visibility; not autoregressive validation",
                sources={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                         ("benchmarks/profile_adaptive_real_activations.py", "benchmarks/profile_real_llm_gate1_heads.py")},
                records=records), captures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max-seq", type=int, default=2048)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--captures", type=Path, help="Replay a previously saved capture archive on CPU")
    p.add_argument("--last-query-only", action="store_true")
    args = p.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.captures:
        torch.set_num_threads(4)
        captures = torch.load(args.captures, map_location="cpu", weights_only=True)
        records = []
        for item in captures:
            q, positions = item["q"], item["query_positions"]
            if args.last_query_only:
                q, positions = q[:, -1:], positions[-1:]
            row = evaluate(q, item["k"], item["v"], positions)
            row.update(prompt_id=item["prompt_id"], layer=item["layer"])
            records.append(row)
        result = dict(schema="streamattn.adaptive_real_physical_replay.v1", complete=True,
                      diagnostic_only=True, performance_promotion=False,
                      capture_sha256=hashlib.sha256(args.captures.read_bytes()).hexdigest(),
                      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      last_query_only=args.last_query_only, records=records)
    else:
        if args.last_query_only:
            p.error("--last-query-only requires --captures")
        result, captures = capture_and_evaluate(args.model, args.max_seq)
        torch.save(captures, args.output_json.with_suffix(".captures.pt"))
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"REAL complete={result['complete']} captures={len(captures)}", flush=True)


if __name__ == "__main__":
    main()
