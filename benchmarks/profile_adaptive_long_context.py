"""One bounded fresh-Qwen-32K feasibility comparison, not a live selector."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.profile_adaptive_feasibility import frontier, native_headroom
from benchmarks.profile_adaptive_real_activations import capture_and_evaluate, evaluate

SCHEMA = "streamattn.adaptive_long_context_feasibility.v1"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
LAYERS = (0, 16, 24, 26, 27)
PROMPTS = ("technical", "instruction_holdout_20260910")
SOURCE_FILES = (
    "benchmarks/profile_adaptive_long_context.py",
    "benchmarks/profile_adaptive_feasibility.py",
    "benchmarks/profile_adaptive_real_activations.py",
    "benchmarks/profile_real_llm_gate1_heads.py",
    "benchmarks/run_lightning_adaptive_two_gate.py",
    "tests/test_adaptive_long_context.py",
    "docs/adaptive_long_context_protocol.md",
)


def validate_capture_contract(metadata, captures):
    if (metadata["model"] != MODEL or metadata["model_revision"] != REVISION
            or metadata["max_seq"] != 32768):
        raise ValueError("fresh capture must use the pinned Qwen 32K configuration")
    identities = [(c["prompt_id"], c["layer"]) for c in captures]
    expected = {(p, layer) for p in PROMPTS for layer in LAYERS}
    if len(identities) != 10 or set(identities) != expected:
        raise ValueError("ten distinct prompt/layer captures required")
    for c in captures:
        if (c["q"].shape != (1, 1, 16, 128) or c["k"].shape != (1, 32768, 2, 128)
                or c["v"].shape != c["k"].shape
                or any(c[name].dtype != torch.bfloat16 for name in ("q", "k", "v"))
                or c["query_positions"].tolist() != [32767]
                or not c["meta"]["rope_applied"]):
            raise ValueError("capture violates B1/M1/BF16/G8/post-RoPE/full-32K contract")


def append_current_certificate(q, k, v, diagnostic, full, references, current):
    """Replay the current centered two-gate support, with its decisions excluded."""
    source = next(r for r in current["results"]
                  if r["radius"] == "mean_centered" and r["decision"] == "two_gate")
    kept = source["kept_blocks"]
    n, hk, dim = k.shape[1:]
    group = diagnostic["group_size"]
    mask = torch.zeros(hk, n, dtype=torch.bool)
    for head, blocks in enumerate(kept):
        for block in blocks:
            mask[head, block * 32:min((block + 1) * 32, n)] = True
    kh = k[0].permute(1, 0, 2).double().repeat_interleave(group, 0)
    vh = v[0].permute(1, 0, 2).double().repeat_interleave(group, 0)
    scores = torch.einsum("hd,hnd->hn", q[0, 0].double(), kh) / dim**0.5
    weights = scores.masked_fill(~mask.repeat_interleave(group, 0), -torch.inf).softmax(-1)
    selected = torch.einsum("hn,hnd->hd", weights, vh)
    error = float((selected - full).norm(dim=-1).max())
    if not torch.isfinite(selected).all() or error > diagnostic["budget"] + 1e-10:
        raise AssertionError("current certificate schedule violates cumulative budget")
    diagnostic["schedules"].append(dict(method="current_two_gate", kept_blocks=kept,
        retained_tokens_per_kv_head=mask.sum(-1).tolist(),
        omitted_token_fraction=1 - float(mask.sum()) / (hk * n),
        max_output_l2_error=error, max_bound=source["max_bound"],
        full_information=True, deployable_selector=False, globally_optimal=False,
        certified_schedule=True, certificate="offline_current_centered_two_gate",
        caveat="Known-support replay omits K and V; a live post-QK decision has already read K. "
               "Decision/update costs are excluded; this is optimistic schedule headroom."))
    references["current_two_gate"] = selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--capture-remote", required=True)
    p.add_argument("--teamspace-id", required=True)
    p.add_argument("--cloud-account", required=True)
    args = p.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    torch.set_num_threads(4)
    metadata, captures = capture_and_evaluate(MODEL, 32768, revision=REVISION, layers=LAYERS,
                                              capture_only=True, last_query_only=True)
    validate_capture_contract(metadata, captures)
    archive = args.output_json.with_suffix(".captures.pt")
    torch.save(captures, archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    from lightning_sdk.api import TeamspaceApi
    from benchmarks.run_lightning_adaptive_two_gate import upload_capture
    upload_capture(archive, api=TeamspaceApi(), teamspace_id=args.teamspace_id,
                   cloud_account=args.cloud_account, remote_path=args.capture_remote)
    result = dict(schema=SCHEMA, complete=False, diagnostic_only=True, performance_promotion=False,
        model=MODEL, model_revision=metadata["model_revision"], max_seq=32768,
        capture_remote_path=args.capture_remote, capture_archive_sha256=digest,
        capture_metadata=metadata, device=torch.cuda.get_device_name(), torch=torch.__version__,
        conditioning="fresh dense-conditioned model forward; final prefill query; not generation",
        contract=dict(batch=1, query_length=1, full_gqa_group=8, logical_block_size=32,
            cumulative_output_l2_budget=1e-3, native_component_gate=0.02,
            native_total_error_certified=False, repetitive_controlled_prompts=True),
        protocol=dict(buffer_copies=8, calls_per_graph=8, iterations=100, trials=7,
            conditions=["warm_fixed_buffer", "rotating_working_set"],
            baseline_winner_resolved_separately=True, preparation_and_decisions_included=False,
            hardware_memory_traffic_measured=False, broader_context_escalation_authorized=False),
        sources={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in SOURCE_FILES},
        records=[])

    def checkpoint():
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)

    checkpoint()
    # Reload outside inference_mode so native metadata has ordinary version counters.
    del captures
    gc.collect()
    torch.cuda.empty_cache()
    captures = torch.load(archive, map_location="cpu", weights_only=True)
    for capture in captures:
        q, k, v, positions = [capture[name] for name in ("q", "k", "v", "query_positions")]
        diagnostic, full, references = frontier(q, k, v, positions, budget=1e-3, block_size=32)
        current = evaluate(q, k, v, positions, budget=1e-3, block_size=32,
                           head_groups=(8,), query_tiles=(1,), include_schedules=True)
        append_current_certificate(q, k, v, diagnostic, full, references, current)
        print(json.dumps(dict(stage="offline32k", prompt_id=capture["prompt_id"], layer=capture["layer"],
            omitted={r["method"]: r["omitted_token_fraction"] for r in diagnostic["schedules"]})), flush=True)
        with torch.no_grad():
            native = native_headroom(q, k, v, diagnostic, full, references,
                                     iterations=100, trials=7, buffer_copies=8)
        result["records"].append(dict(prompt_id=capture["prompt_id"], layer=capture["layer"],
            **diagnostic, current_certificate=current["results"], native=native))
        checkpoint()
        gc.collect()
        torch.cuda.empty_cache()
    result["complete"] = True
    checkpoint()


if __name__ == "__main__":
    main()
