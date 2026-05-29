import json

import pytest

from benchmarks.profile_gate0_seed_only_multi_layer_rollout import RouteBundle
from benchmarks.profile_gate0_seed_only_multi_layer_rollout import _validate_route_bundle
from benchmarks.profile_gate0_seed_only_closed_loop_rollout import _prompt_rows_from_file
from benchmarks.profile_gate0_seed_only_closed_loop_rollout import _prompts_from_args
from benchmarks.build_seed_policy_stress_prompts import BUCKETS, build_rows
from benchmarks.profile_seed_only_route_bundle_decode import (
    StreamAttnNativeKVCache,
    StreamAttnQwenAttentionModule,
    _SeedOnlyQwenDecodePatch,
    _apply_product_fast_path_args,
    _batch_tokens,
    _bucket_route_policy_decision,
    _native_cache_from_hf_cache,
    _native_cache_mask_bookkeeping,
    _parent_module_and_attr,
    apply_layer_seed_overrides,
    apply_policy_kernel_tunable_overrides,
    parse_layer_id_set,
    parse_layer_kernel_tunable_overrides,
    parse_layer_seed_overrides,
    summarize_patch_timing_rows,
)
from benchmarks.summarize_seed_policy_stress_replay import summarize_payload
from benchmarks.profile_seed_only_stress_attribution import (
    RouteSpec,
    build_route_specs,
    failure_score,
)
from benchmarks.profile_seed_policy_attention_coverage import (
    _capture_metrics_for_head,
    _first_cache_like,
    _first_cache_position_like,
    _first_positional_tensor_pair,
    _js_divergence,
    _parse_selector_profiles,
    _row_recommendation,
    _seed_indices,
    _selector_seed_masks,
    _selected_rows_from_artifact,
    _stress_terms,
)
from benchmarks.profile_seed_policy_repair_sweep import (
    STRICT_LAYERS,
    build_bucket_prompt_pack,
    build_repair_variants,
)
from benchmarks.summarize_seed_policy_attention_coverage import (
    _metric_summary as attention_coverage_metric_summary,
)
from stream_attention.decode import Gate0SeedOnlyBatchedPolicy
from stream_attention.seed_selectors import (
    select_seed_blocks_by_profile,
    seed_token_indices_from_blocks,
)
from stream_attention.kernels.gate0_seed_only_triton import (
    gate0_refresh_packed_seed_cache_recent_bhsd,
    gate0_seed_only_packed_ring_append_triton_forward_out,
    make_gate0_seed_only_packed_workspace,
)


def test_parse_layer_seed_overrides_named_fields():
    assert parse_layer_seed_overrides("2:sink=2,recent=4,middle=10,block=32") == {
        2: {
            "sink_blocks": 2,
            "recent_blocks": 4,
            "middle_seed_blocks": 10,
            "block_size": 32,
        }
    }


def test_parse_layer_seed_overrides_positional_fields():
    assert parse_layer_seed_overrides("2:2,4,10;18:32,2,6,12") == {
        2: {
            "sink_blocks": 2,
            "recent_blocks": 4,
            "middle_seed_blocks": 10,
        },
        18: {
            "block_size": 32,
            "sink_blocks": 2,
            "recent_blocks": 6,
            "middle_seed_blocks": 12,
        },
    }


def test_parse_layer_kernel_tunable_overrides_named_and_positional_fields():
    assert parse_layer_kernel_tunable_overrides("35:warps=8,stages=2;0:4,1") == {
        35: {"num_warps": 8, "num_stages": 2},
        0: {"num_warps": 4, "num_stages": 1},
    }


def test_parse_layer_id_set_accepts_commas_and_semicolons():
    assert parse_layer_id_set("") == set()
    assert parse_layer_id_set("0,14;16") == {0, 14, 16}


def test_product_fast_path_applies_validated_qwen3b_runtime_flags():
    import argparse

    args = argparse.Namespace(
        product_fast_path="qwen25_3b_b8_validated",
        model="Qwen/Qwen2.5-3B-Instruct",
        batch_size=8,
        max_seq=32768,
        dynamic_selector_profile="",
        dynamic_selector_layers="",
        bucket_route_policy="none",
        product_strict=False,
        use_packaged_policies=False,
        native_routed_cache=False,
        fused_rope_append_seed=False,
        packed_qkv_projection=False,
        packed_qkv_fused_input=True,
        native_attention_module=True,
        direct_o_proj=False,
        triton_o_proj=True,
        layer_kernel_overrides="",
        allow_mixed_seed_configs=False,
    )

    _apply_product_fast_path_args(args)

    assert args.bucket_route_policy == "qwen25_3b_b8"
    assert args.product_strict is True
    assert args.use_packaged_policies is True
    assert args.native_routed_cache is True
    assert args.fused_rope_append_seed is True
    assert args.packed_qkv_projection is True
    assert args.packed_qkv_fused_input is False
    assert args.native_attention_module is False
    assert args.direct_o_proj is True
    assert args.triton_o_proj is False
    assert args.layer_kernel_overrides == (
        "0:warps=8,stages=2;27:warps=8,stages=2;35:warps=8,stages=2"
    )
    assert args.allow_mixed_seed_configs is True


def test_prompt_rows_from_jsonl_preserves_stress_metadata(tmp_path):
    import json

    path = tmp_path / "stress.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "code_000",
                "bucket": "code",
                "language": "python",
                "risk": "identifier",
                "prompt": "alpha",
            }
        )
        + "\n"
        + json.dumps({"id": "math_000", "bucket": "math", "text": "beta"})
        + "\n",
        encoding="utf-8",
    )

    rows = _prompt_rows_from_file(str(path), max_rows=1)

    assert rows == [
        {
            "kind": "code",
            "prompt": "alpha",
            "id": "code_000",
            "bucket": "code",
            "language": "python",
            "risk": "identifier",
        }
    ]


def test_prompt_rows_from_file_filters_rows_per_kind(tmp_path):
    import json

    path = tmp_path / "prompts.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"bucket": "a", "prompt": "a0"}),
                json.dumps({"bucket": "a", "prompt": "a1"}),
                json.dumps({"bucket": "b", "prompt": "b0"}),
                json.dumps({"bucket": "b", "prompt": "b1"}),
                json.dumps({"bucket": "c", "prompt": "c0"}),
            ]
        ),
        encoding="utf-8",
    )

    rows = _prompt_rows_from_file(
        str(path),
        max_rows=4,
        prompt_file_kinds="a,b",
        rows_per_kind=1,
    )

    assert [(row["bucket"], row["prompt"]) for row in rows] == [("a", "a0"), ("b", "b0")]


def test_prompts_from_args_cycles_prompt_kinds_to_batch_size():
    import argparse

    rows = _prompts_from_args(
        argparse.Namespace(
            prompt_file="",
            prompt_kinds="needle,code",
            batch_size=5,
            prompt_repeat=1,
        )
    )

    assert [row["kind"] for row in rows] == ["needle", "code", "needle", "code", "needle"]


def test_build_seed_policy_stress_prompts_covers_all_buckets():
    rows = build_rows(rows_per_bucket=1, target_words=1200)

    assert [row["bucket"] for row in rows] == list(BUCKETS)
    assert all(row["prompt"] for row in rows)
    assert {row["kind"] for row in rows} == set(BUCKETS)


def test_stress_replay_summarizer_reports_worst_bucket(tmp_path):
    import json

    path = tmp_path / "stress.json"
    path.write_text(
        json.dumps(
            {
                "decision": {"passed": False},
                "route_bundle": {"layers": [0, 2]},
                "timing": {
                    "speedup_vs_dense_decode": 1.2,
                    "dense_decode_ms_per_token": 30.0,
                    "streamattn_decode_ms_per_token": 25.0,
                },
                "safety": {
                    "case_count": 16,
                    "kl_max": 0.2,
                    "kl_p99": 0.1,
                    "top1_changed_count": 0,
                    "sample_token_changed_count": 0,
                    "topk_overlap_min": 3,
                    "reference_top1_logprob_delta_max_abs": 0.01,
                    "by_prompt_bucket": {
                        "code": {"kl_max": 0.01, "kl_p99": 0.01, "topk_overlap_min": 4},
                        "needle": {
                            "kl_max": 0.2,
                            "kl_p99": 0.1,
                            "topk_overlap_min": 3,
                            "reference_top1_logprob_delta_max_abs": 0.01,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_payload(path)

    assert summary["passed"] is False
    assert summary["speedup_vs_dense_decode"] == 1.2
    assert summary["worst_bucket"]["bucket"] == "needle"


def test_stress_attribution_route_specs_include_leaveouts_and_candidates():
    specs = build_route_specs(
        base_layers=[0, 14, 16],
        route_set="full",
        add_layers=[2, 29],
    )
    by_name = {spec.name: spec.layers for spec in specs}

    assert by_name["strict_base"] == (0, 14, 16)
    assert by_name["minus_l14"] == (0, 16)
    assert by_name["single_l16"] == (16,)
    assert by_name["plus_l2"] == (0, 2, 14, 16)
    assert by_name["single_l29"] == (29,)


def test_stress_attribution_policy_names_and_mixed_seed_detection():
    strict = RouteSpec("strict", (0, 14))
    mixed = RouteSpec("mixed", (0, 2, 14))

    assert strict.policy_names == (
        "qwen25_3b_l0_32k_seed_only_batched,"
        "qwen25_3b_l14_32k_seed_only_batched"
    )
    assert strict.allow_mixed_seed_configs is False
    assert mixed.allow_mixed_seed_configs is True


def test_stress_attribution_failure_score_weights_failures():
    safe = {
        "kl_max": 1.0e-5,
        "target_logprob_delta_max_abs": 1.0e-4,
        "top1_changes": 0,
        "sample_changes": 0,
        "topk_overlap_min": 5,
    }
    unsafe = {
        "kl_max": 2.0e-4,
        "target_logprob_delta_max_abs": 3.0e-3,
        "top1_changes": 2,
        "sample_changes": 1,
        "topk_overlap_min": 3,
    }

    assert failure_score(safe) == 0.0
    assert failure_score(unsafe) > 35.0


def test_attention_coverage_seed_indices_recent_first():
    seed = _seed_indices(
        seq_len=320,
        block_size=32,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=2,
        block_order="recent_first",
    )

    assert seed.numel() == 128
    assert seed[:3].tolist() == [0, 1, 2]
    assert 224 in seed.tolist()
    assert 255 in seed.tolist()
    assert 256 in seed.tolist()
    assert 319 in seed.tolist()


def test_attention_coverage_stress_terms_noisy_neartie():
    support, distractors = _stress_terms({"bucket": "noisy_neartie", "id": "noisy_neartie_001"})

    assert any("azure" in term for term in support)
    assert any("amber" in term for term in distractors)


def test_attention_coverage_selected_rows_from_artifact(tmp_path):
    artifact = {
        "prompts": [
            {"row": 0, "bucket": "code"},
            {"row": 1, "bucket": "chat_instruction"},
        ],
        "safety": {
            "by_prompt_bucket": {
                "chat_instruction": {
                    "first_divergence": {"rows": [1], "step": 3},
                    "worst_case_by_kl": {"row": 1},
                },
                "code": {"first_divergence": {"rows": [0], "step": 2}},
            }
        },
    }
    path = tmp_path / "stress.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    selected = _selected_rows_from_artifact(str(path), target_buckets={"chat_instruction"}, include_step0=True)

    assert selected == {1: {0, 3}}


def test_attention_coverage_metrics_and_recommendation():
    import torch

    q = torch.tensor([1.0, 0.0])
    k = torch.tensor([[4.0, 0.0], [0.0, 1.0], [3.0, 0.0]])
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0], [10.0, 0.0]])
    seed_mask = torch.tensor([True, True, False])
    support_mask = torch.tensor([False, False, True])
    distractor_mask = torch.tensor([True, False, False])

    metrics, probs = _capture_metrics_for_head(
        q=q,
        k=k,
        v=v,
        seed_mask=seed_mask,
        support_mask=support_mask,
        distractor_mask=distractor_mask,
    )

    assert probs.shape == (3,)
    assert metrics["support_out_seed"] > 0.0
    assert metrics["mass_seed"] + metrics["mass_omitted"] == pytest.approx(1.0)
    assert _row_recommendation(metrics) in {"coverage_repair", "value_sensitive_repair"}


def test_attention_coverage_selector_profiles_reject_unknown():
    assert _parse_selector_profiles("fixed_policy,qk_block_max") == ["fixed_policy", "qk_block_max"]

    with pytest.raises(ValueError, match="unknown selector profiles"):
        _parse_selector_profiles("fixed_policy,not_a_selector")


def test_attention_coverage_support_selector_adds_support_block():
    import torch

    policy = Gate0SeedOnlyBatchedPolicy(
        policy_id="p0",
        model_id="m",
        layer_id=0,
        block_size=4,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=1,
        block_order="recent_first",
    )
    scores = torch.zeros(16)
    probs = torch.full((16,), 1.0 / 16.0)
    q = torch.tensor([1.0, 0.0])
    k = torch.ones((16, 2))
    v = torch.ones((16, 2))
    support = torch.zeros(16, dtype=torch.bool)
    support[8:12] = True
    distractor = torch.zeros(16, dtype=torch.bool)

    masks = _selector_seed_masks(
        q=q,
        k=k,
        scores=scores,
        probs=probs,
        v=v,
        support_mask=support,
        distractor_mask=distractor,
        policy=policy,
        selector_profiles=["fixed_policy", "support_block_oracle"],
    )

    assert masks["fixed_policy"]["selected_blocks"] == [0, 3, 2]
    assert masks["support_block_oracle"]["selected_blocks"] == [0, 3, 2]

    support[4:8] = True
    masks = _selector_seed_masks(
        q=q,
        k=k,
        scores=scores,
        probs=probs,
        v=v,
        support_mask=support,
        distractor_mask=distractor,
        policy=policy,
        selector_profiles=["support_block_oracle"],
    )

    assert masks["support_block_oracle"]["selected_blocks"] == [0, 3, 1]


def test_attention_coverage_qk_selector_keeps_negative_score_ordering():
    import torch

    policy = Gate0SeedOnlyBatchedPolicy(
        policy_id="p0",
        model_id="m",
        layer_id=0,
        block_size=4,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=1,
        block_order="recent_first",
    )
    q = torch.tensor([1.0, 0.0])
    k = -torch.ones((16, 2))
    v = torch.ones((16, 2))
    scores = torch.full((16,), -10.0)
    scores[4] = -1.0
    probs = torch.softmax(scores, dim=0)
    support = torch.zeros(16, dtype=torch.bool)
    distractor = torch.zeros(16, dtype=torch.bool)

    masks = _selector_seed_masks(
        q=q,
        k=k,
        scores=scores,
        probs=probs,
        v=v,
        support_mask=support,
        distractor_mask=distractor,
        policy=policy,
        selector_profiles=["qk_block_max"],
    )

    assert masks["qk_block_max"]["selected_blocks"] == [0, 3, 1]


def test_attention_coverage_support_extreme_selector_is_query_aware():
    import torch

    policy = Gate0SeedOnlyBatchedPolicy(
        policy_id="p0",
        model_id="m",
        layer_id=0,
        block_size=4,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=1,
        block_order="recent_first",
    )
    q = torch.tensor([1.0, 0.0])
    k = torch.zeros((16, 2))
    k[4] = torch.tensor([5.0, 0.0])
    k[5] = torch.tensor([-5.0, 0.0])
    v = torch.ones((16, 2))
    scores = k @ q
    probs = torch.softmax(scores, dim=0)
    support = torch.zeros(16, dtype=torch.bool)
    distractor = torch.zeros(16, dtype=torch.bool)

    masks = _selector_seed_masks(
        q=q,
        k=k,
        scores=scores,
        probs=probs,
        v=v,
        support_mask=support,
        distractor_mask=distractor,
        policy=policy,
        selector_profiles=["support_extreme2_mean", "support_extreme2_mean_refine32"],
    )

    assert masks["support_extreme2_mean"]["selected_blocks"] == [0, 3, 1]
    assert masks["support_extreme2_mean_refine32"]["selected_blocks"] == [0, 3, 1]
    assert masks["support_extreme2_mean"]["selector_estimated_dot_token_ratio"] == pytest.approx(0.5)
    assert masks["support_extreme2_mean_refine32"]["selector_estimated_dot_tokens"] == pytest.approx(24.0)


def test_attention_coverage_random_support_selector_is_registered():
    assert _parse_selector_profiles("support_rand4,support_rand8_refine32") == [
        "support_rand4",
        "support_rand8_refine32",
    ]


def test_runtime_support_rand_refine_selector_finds_aligned_block():
    import torch

    policy = Gate0SeedOnlyBatchedPolicy(
        policy_id="p0",
        model_id="m",
        layer_id=0,
        block_size=4,
        sink_blocks=1,
        recent_blocks=1,
        middle_seed_blocks=1,
        block_order="recent_first",
    )
    q = torch.tensor([1.0, 0.0, 0.0, 0.0])
    k = torch.zeros((16, 4))
    k[8] = torch.tensor([9.0, 0.0, 0.0, 0.0])
    k[4] = torch.tensor([2.0, 0.0, 0.0, 0.0])

    selection = select_seed_blocks_by_profile(
        q=q,
        k=k,
        policy=policy,
        selector="support_rand8_refine32",
    )

    assert selection.selected_blocks == [0, 3, 2]
    assert selection.middle_blocks == [2]
    assert selection.cost["selector_proxy_dot_tokens"] > 0.0
    assert selection.cost["selector_refine_dot_tokens"] > 0.0


def test_seed_token_indices_from_blocks_deduplicates_and_sorts():
    import torch

    indices = seed_token_indices_from_blocks(
        seq_len=10,
        block_size=4,
        blocks=[2, 0, 2],
        device=torch.device("cpu"),
    )

    assert indices.tolist() == [0, 1, 2, 3, 8, 9]


def test_attention_coverage_hook_input_inference():
    import torch

    class DummyCache:
        pass

    cache = DummyCache()
    cache.layers = []
    pair = (torch.ones(1), torch.zeros(1))
    cache_pos = torch.tensor([7], dtype=torch.long)

    assert _first_positional_tensor_pair([None, pair]) == pair
    assert _first_cache_like([None, torch.ones(1), cache]) is cache
    assert _first_cache_position_like([torch.ones((1, 2)), cache_pos]) is cache_pos


def test_attention_coverage_js_and_summary():
    import torch

    p = torch.tensor([0.8, 0.2])
    q = torch.tensor([0.5, 0.5])
    assert _js_divergence(p, p) == pytest.approx(0.0)
    assert _js_divergence(p, q) > 0.0

    summary = attention_coverage_metric_summary(
        [
            {
                "mass_omitted": 0.3,
                "support_out_seed": 0.1,
                "delta_collapse": 0.05,
                "value_residual_ratio": 0.2,
                "dense_vs_route_attention_js": 0.01,
            }
        ]
    )
    assert summary["top_recommendation"] == "coverage_repair"


def test_attention_coverage_summarizer_reports_selector_groups(tmp_path):
    from benchmarks.summarize_seed_policy_attention_coverage import summarize_payload

    path = tmp_path / "coverage.json"
    path.write_text(
        json.dumps(
            {
                "model": "m",
                "target_layers": [26],
                "routed_layers": [26],
                "target_buckets": ["json_tool"],
                "selector_profiles": ["fixed_policy", "qk_block_max"],
                "capture_steps": [0],
                "rows": [
                    {
                        "selector": "fixed_policy",
                        "layer": 26,
                        "bucket": "json_tool",
                        "condition": "dense_conditioned",
                        "mass_omitted": 0.5,
                        "support_out_seed": 0.2,
                        "delta_collapse": 0.2,
                        "value_residual_ratio": 0.4,
                        "dense_vs_route_attention_js": 0.0,
                    },
                    {
                        "selector": "qk_block_max",
                        "layer": 26,
                        "bucket": "json_tool",
                        "condition": "dense_conditioned",
                        "mass_omitted": 0.2,
                        "support_out_seed": 0.1,
                        "delta_collapse": 0.1,
                        "value_residual_ratio": 0.2,
                        "dense_vs_route_attention_js": 0.0,
                        "selector_estimated_dot_tokens": 16.0,
                        "selector_estimated_dot_token_ratio": 1.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_payload(path)

    assert summary["selector_profiles"] == ["fixed_policy", "qk_block_max"]
    assert {row["selector"] for row in summary["by_selector"]} == {"fixed_policy", "qk_block_max"}
    qk_summary = next(row for row in summary["by_selector"] if row["selector"] == "qk_block_max")
    assert qk_summary["selector_estimated_dot_token_ratio_mean"] == pytest.approx(1.0)


def test_repair_sweep_builds_focused_variants():
    variants = {variant.name: variant for variant in build_repair_variants("focused")}

    assert variants["strict_base"].layers == STRICT_LAYERS
    assert 27 not in variants["minus_l27"].layers
    assert 26 not in variants["minus_l26_l27"].layers
    assert 27 not in variants["minus_l26_l27"].layers
    assert "26:" in variants["l26_l27_s640"].overrides
    assert "27:" in variants["l26_l27_s640"].overrides


def test_repair_sweep_prompt_pack_repeats_target_buckets(tmp_path):
    rows = [
        {"id": "chat_0", "bucket": "chat_instruction", "kind": "chat_instruction", "prompt": "chat"},
        {"id": "json_0", "bucket": "json_tool", "kind": "json_tool", "prompt": "json"},
        {"id": "code_0", "bucket": "code", "kind": "code", "prompt": "code"},
    ]
    source = tmp_path / "prompts.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "repair.jsonl"

    selected = build_bucket_prompt_pack(
        source_prompt_file=source,
        target_buckets=["chat_instruction", "json_tool"],
        batch_size=5,
        output_path=output,
    )

    assert [row["bucket"] for row in selected] == [
        "chat_instruction",
        "json_tool",
        "chat_instruction",
        "json_tool",
        "chat_instruction",
    ]
    assert selected[0]["repair_source_id"] == "chat_0"
    assert output.exists()


def test_route_bundle_bucket_policy_uses_full_route_for_validated_batch():
    decision = _bucket_route_policy_decision(
        [{"bucket": "code"}, {"bucket": "long_doc"}],
        policy_name="qwen25_3b_b8",
        product_strict=True,
    )

    assert decision["batch_mode"] == "seed_only_bundle"
    assert decision["seed_only_layers"] == [0, 2, 14, 16, 24, 26, 27, 35]
    assert "qwen25_3b_l2_s416_32k_seed_only_batched" in decision["policy_names"]
    assert "qwen25_3b_l27_32k_seed_only_batched" in decision["policy_names"]


def test_route_bundle_bucket_policy_fails_closed_for_stress_bucket():
    decision = _bucket_route_policy_decision(
        [{"bucket": "json_tool"}],
        policy_name="qwen25_3b_b8",
        product_strict=True,
    )

    assert decision["batch_mode"] == "exact_native"
    assert decision["fallback_reason"] == "batch_contains_exact_bucket"
    assert decision["policy_names"] == []


def test_route_bundle_bucket_policy_fails_closed_for_mixed_batch():
    decision = _bucket_route_policy_decision(
        [{"bucket": "code"}, {"bucket": "json_tool"}],
        policy_name="qwen25_3b_b8",
        product_strict=True,
    )

    assert decision["batch_mode"] == "exact_native"
    assert any("json_tool" in item for item in decision["fallback_details"])


def test_route_bundle_bucket_policy_research_reduced_route_for_json_tool():
    decision = _bucket_route_policy_decision(
        [{"bucket": "json_tool"}],
        policy_name="qwen25_3b_b8",
        product_strict=False,
    )

    assert decision["batch_mode"] == "seed_only_bundle"
    assert decision["seed_only_layers"] == [0, 14, 16, 24, 35]
    assert "qwen25_3b_l26_32k_seed_only_batched" not in decision["policy_names"]
    assert "qwen25_3b_l27_32k_seed_only_batched" not in decision["policy_names"]


def test_route_bundle_bucket_policy_unknown_bucket_fails_closed():
    decision = _bucket_route_policy_decision(
        [{"bucket": "new_task"}],
        policy_name="qwen25_3b_b8",
        product_strict=True,
    )

    assert decision["batch_mode"] == "exact_native"
    assert any("bucket_not_validated" in item for item in decision["fallback_details"])


def test_batch_tokens_temporarily_sets_truncation_side():
    import torch

    class Encoded(dict):
        def to(self, _device):
            return self

    class Tokenizer:
        truncation_side = "right"

        def __init__(self):
            self.calls = []

        def __call__(self, texts, **kwargs):
            self.calls.append((self.truncation_side, list(texts), kwargs["max_length"]))
            return Encoded(
                {
                    "input_ids": torch.ones((1, kwargs["max_length"]), dtype=torch.long),
                    "attention_mask": torch.ones((1, kwargs["max_length"]), dtype=torch.long),
                }
            )

    tokenizer = Tokenizer()

    _batch_tokens(
        tokenizer,
        [{"kind": "stress", "prompt": "alpha"}],
        max_seq=4,
        device=torch.device("cpu"),
        truncation_side="left",
    )

    assert tokenizer.calls == [("left", ["alpha"], 4)]
    assert tokenizer.truncation_side == "right"


def test_native_cache_mask_bookkeeping_overrides_and_restores():
    class DummyLayer:
        def get_seq_length(self, cache_position=None):
            return 4

        def get_mask_sizes(self, cache_position):
            return 4, 0

    class DummyCache:
        def __init__(self):
            self.layers = [DummyLayer()]

        def get_seq_length(self, layer_idx=0, cache_position=None):
            return self.layers[layer_idx].get_seq_length(cache_position)

        def get_mask_sizes(self, cache_position, layer_idx):
            return self.layers[layer_idx].get_mask_sizes(cache_position)

    cache = DummyCache()
    original_mask = cache.get_mask_sizes
    original_seq = cache.get_seq_length
    original_layer_mask = cache.layers[0].get_mask_sizes
    original_layer_seq = cache.layers[0].get_seq_length
    with _native_cache_mask_bookkeeping(cache, enabled=True):
        assert cache.get_mask_sizes(None, 0) == (4, 0)
        assert cache.get_seq_length(0) == 4
        assert cache.layers[0].get_mask_sizes(None) == (4, 0)
        assert cache.layers[0].get_seq_length() == 4
        cache._streamattn_past_kv_length = 8
        cache._streamattn_mask_kv_length = 9
        assert cache.get_mask_sizes(None, 0) == (9, 0)
        assert cache.get_seq_length(0) == 8
        assert cache.layers[0].get_mask_sizes(None) == (9, 0)
        assert cache.layers[0].get_seq_length() == 8

    assert cache.get_mask_sizes(None, 0) == (4, 0)
    assert cache.get_seq_length(0) == 4
    assert cache.layers[0].get_mask_sizes(None) == (4, 0)
    assert cache.layers[0].get_seq_length() == 4
    assert not hasattr(cache, "_streamattn_mask_kv_length")
    assert not hasattr(cache, "_streamattn_past_kv_length")
    assert cache.get_mask_sizes.__func__ is original_mask.__func__
    assert cache.get_seq_length.__func__ is original_seq.__func__
    assert cache.layers[0].get_mask_sizes.__func__ is original_layer_mask.__func__
    assert cache.layers[0].get_seq_length.__func__ is original_layer_seq.__func__


def test_apply_layer_seed_overrides_rewrites_only_target_layer():
    policies = [
        Gate0SeedOnlyBatchedPolicy(
            policy_id="p0",
            model_id="m",
            layer_id=0,
            dtype="fp16",
            kv_len_bucket=32768,
            heads=16,
            kv_heads=2,
            dim=128,
        ),
        Gate0SeedOnlyBatchedPolicy(
            policy_id="p2",
            model_id="m",
            layer_id=2,
            dtype="fp16",
            kv_len_bucket=32768,
            heads=16,
            kv_heads=2,
            dim=128,
        ),
    ]
    bundle = RouteBundle(
        policy_names=["p0", "p2"],
        policies=policies,
        artifacts=[{}, {}],
        layer_ids=[0, 2],
    )

    updated, summaries = apply_layer_seed_overrides(
        bundle,
        "2:sink=2,recent=4,middle=10,block=32",
    )

    assert updated.layer_ids == [0, 2]
    assert updated.policies[0].recent_blocks == 2
    assert updated.policies[1].recent_blocks == 4
    assert updated.policies[1].middle_seed_blocks == 10
    assert updated.policies[1].policy_id == "p2_s32_2_4_10"
    assert summaries[0]["layer_id"] == 2
    assert summaries[0]["new"]["seed_tokens"] == 512


def test_apply_policy_kernel_tunable_overrides_rewrites_packaged_layers():
    policies = [
        Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0),
        Gate0SeedOnlyBatchedPolicy(policy_id="p2", model_id="m", layer_id=2),
    ]
    bundle = RouteBundle(
        policy_names=["p0", "p2"],
        policies=policies,
        artifacts=[{}, {}],
        layer_ids=[0, 2],
    )

    updated, summaries = apply_policy_kernel_tunable_overrides(
        bundle,
        num_warps=2,
        num_stages=1,
        enabled=True,
    )

    assert updated.layer_ids == [0, 2]
    assert [policy.num_warps for policy in updated.policies] == [2, 2]
    assert [policy.num_stages for policy in updated.policies] == [1, 1]
    assert [policy.policy_id for policy in updated.policies] == ["p0_w2_s1", "p2_w2_s1"]
    assert summaries[0]["old"]["num_warps"] == 4
    assert summaries[0]["new"]["num_stages"] == 1


def test_apply_policy_kernel_tunable_overrides_rewrites_selected_layers():
    policies = [
        Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0),
        Gate0SeedOnlyBatchedPolicy(policy_id="p35", model_id="m", layer_id=35),
    ]
    bundle = RouteBundle(
        policy_names=["p0", "p35"],
        policies=policies,
        artifacts=[{}, {}],
        layer_ids=[0, 35],
    )

    updated, summaries = apply_policy_kernel_tunable_overrides(
        bundle,
        num_warps=4,
        num_stages=2,
        enabled=False,
        layer_override_text="35:warps=8,stages=2",
    )

    assert updated.policies[0].policy_id == "p0"
    assert updated.policies[0].num_warps == 4
    assert updated.policies[1].policy_id == "p35_w8_s2"
    assert updated.policies[1].num_warps == 8
    assert [item["layer_id"] for item in summaries] == [35]


def test_validate_route_bundle_allows_explicit_mixed_seed_configs():
    import argparse
    import pytest

    policies = [
        Gate0SeedOnlyBatchedPolicy(
            policy_id="p0",
            model_id="m",
            layer_id=0,
            dtype="fp16",
            kv_len_bucket=32768,
            min_batch=8,
            heads=16,
            kv_heads=2,
            dim=128,
            recent_blocks=2,
            middle_seed_blocks=8,
        ),
        Gate0SeedOnlyBatchedPolicy(
            policy_id="p2_s640",
            model_id="m",
            layer_id=2,
            dtype="fp16",
            kv_len_bucket=32768,
            min_batch=8,
            heads=16,
            kv_heads=2,
            dim=128,
            recent_blocks=6,
            middle_seed_blocks=12,
        ),
    ]
    args = argparse.Namespace(
        model="m",
        dtype="fp16",
        max_seq=32768,
        batch_size=8,
        allow_mixed_seed_configs=False,
    )

    with pytest.raises(ValueError, match="inconsistent seed field"):
        _validate_route_bundle(policies, args=args)

    args.allow_mixed_seed_configs = True
    _validate_route_bundle(policies, args=args)


def test_summarize_patch_timing_rows_reports_stage_shares():
    summary = summarize_patch_timing_rows(
        [
            {
                "qkv_ms": 1.0,
                "rope_ms": 1.0,
                "cache_update_ms": 1.0,
                "layout_ms": 1.0,
                "seed_kernel_ms": 4.0,
                "output_proj_ms": 2.0,
                "total_ms": 10.0,
            },
            {
                "qkv_ms": 2.0,
                "rope_ms": 1.0,
                "cache_update_ms": 1.0,
                "layout_ms": 1.0,
                "seed_kernel_ms": 3.0,
                "output_proj_ms": 2.0,
                "total_ms": 10.0,
            },
        ]
    )

    assert summary["call_count"] == 2
    assert summary["total_ms"]["sum_ms"] == 20.0
    assert summary["stages"]["seed_kernel_ms"]["sum_ms"] == 7.0
    assert summary["stages"]["seed_kernel_ms"]["share_of_patch_total"] == 0.35
    assert summary["stages"]["qkv_ms"]["mean_ms"] == 1.5


def test_seed_only_qwen_patch_reuses_output_buffer():
    import torch

    patch = _SeedOnlyQwenDecodePatch(
        policy=Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0),
        original_forward=lambda *args, **kwargs: None,
    )

    first = patch._output_buffer(
        batch=2,
        q_heads=4,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = patch._output_buffer(
        batch=2,
        q_heads=4,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    resized = patch._output_buffer(
        batch=3,
        q_heads=4,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert second.data_ptr() == first.data_ptr()
    assert resized.shape == (3, 1, 4, 8)
    assert resized.data_ptr() != first.data_ptr()


def test_seed_only_qwen_patch_packed_qkv_matches_separate_projection():
    import torch

    class DummyAttention(torch.nn.Module):
        head_dim = 4

        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=True)
            self.k_proj = torch.nn.Linear(8, 4, bias=True)
            self.v_proj = torch.nn.Linear(8, 4, bias=True)

    module = DummyAttention()
    hidden = torch.randn(2, 1, 8)
    hidden_shape = (2, 1, -1, module.head_dim)

    separate_patch = _SeedOnlyQwenDecodePatch(
        policy=Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0),
        original_forward=lambda *args, **kwargs: None,
    )
    packed_patch = _SeedOnlyQwenDecodePatch(
        policy=Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0),
        original_forward=lambda *args, **kwargs: None,
        packed_qkv_projection=True,
    )
    packed_patch.prepare_packed_qkv(module)

    separate = separate_patch._qkv_projection(module, hidden, hidden_shape)
    packed = packed_patch._qkv_projection(module, hidden, hidden_shape)

    for packed_tensor, separate_tensor in zip(packed, separate):
        torch.testing.assert_close(packed_tensor, separate_tensor)


def test_seed_only_qwen_patch_packed_qkv_prepares_lazily():
    import torch

    class DummyAttention(torch.nn.Module):
        head_dim = 4

        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=True)
            self.k_proj = torch.nn.Linear(8, 4, bias=True)
            self.v_proj = torch.nn.Linear(8, 4, bias=True)

    module = DummyAttention()
    hidden = torch.randn(2, 1, 8)
    hidden_shape = (2, 1, -1, module.head_dim)
    patch = _SeedOnlyQwenDecodePatch(
        policy=Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0),
        original_forward=lambda *args, **kwargs: None,
        packed_qkv_projection=True,
    )

    assert patch._packed_qkv_weight is None
    q, k, v = patch._qkv_projection(module, hidden, hidden_shape)

    assert patch._packed_qkv_weight is not None
    assert q.shape == (2, 2, 1, 4)
    assert k.shape == (2, 1, 1, 4)
    assert v.shape == (2, 1, 1, 4)


def test_seed_only_qwen_patch_direct_o_proj_matches_module():
    import torch

    class DummyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = torch.nn.Linear(8, 8)

    module = DummyAttention()
    attn_output = torch.randn(2, 1, 8)
    module_patch = _SeedOnlyQwenDecodePatch(
        policy=Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0),
        original_forward=lambda *args, **kwargs: None,
        direct_o_proj=False,
    )
    direct_patch = _SeedOnlyQwenDecodePatch(
        policy=Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0),
        original_forward=lambda *args, **kwargs: None,
        direct_o_proj=True,
    )

    torch.testing.assert_close(
        direct_patch._o_projection(module, attn_output),
        module_patch._o_projection(module, attn_output),
    )


def test_seed_only_qwen_patch_prealloc_o_proj_matches_module_and_reuses_buffer():
    import torch

    class DummyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = torch.nn.Linear(8, 8)

    module = DummyAttention()
    attn_output = torch.randn(2, 1, 8)
    module_patch = _SeedOnlyQwenDecodePatch(
        policy=Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0),
        original_forward=lambda *args, **kwargs: None,
    )
    prealloc_patch = _SeedOnlyQwenDecodePatch(
        policy=Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0),
        original_forward=lambda *args, **kwargs: None,
        prealloc_o_proj=True,
    )

    expected = module_patch._o_projection(module, attn_output)
    first = prealloc_patch._o_projection(module, attn_output)
    torch.testing.assert_close(first, expected)
    first_ptr = first.data_ptr()

    second = prealloc_patch._o_projection(module, attn_output + 1.0)
    assert second.data_ptr() == first_ptr
    torch.testing.assert_close(second, module_patch._o_projection(module, attn_output + 1.0))


def test_packed_seed_workspace_shape():
    import torch

    q = torch.empty((2, 1, 4, 8), dtype=torch.float16)

    workspace = make_gate0_seed_only_packed_workspace(q, seed_tokens=96)

    assert workspace["k_seed"].shape == (2, 4, 96, 8)
    assert workspace["v_seed"].shape == (2, 4, 96, 8)
    assert workspace["k_seed"].dtype == q.dtype


def test_refresh_packed_seed_recent_noops_without_recent_blocks():
    import torch

    key = torch.empty((1, 64, 2, 8), dtype=torch.float16)
    value = torch.empty_like(key)
    q = torch.empty((1, 1, 4, 8), dtype=torch.float16)
    workspace = make_gate0_seed_only_packed_workspace(q, seed_tokens=64)

    k_seed, v_seed = gate0_refresh_packed_seed_cache_recent_bhsd(
        key,
        value,
        workspace["k_seed"],
        workspace["v_seed"],
        q_heads=4,
        block_size=32,
        sink_blocks=1,
        recent_blocks=0,
        middle_seed_blocks=1,
    )

    assert k_seed is workspace["k_seed"]
    assert v_seed is workspace["v_seed"]


def test_packed_ring_append_rejects_missing_recent_blocks_without_triton():
    import pytest
    import torch

    q = torch.empty((1, 1, 4, 8), dtype=torch.float16)
    key_current = torch.empty((1, 1, 2, 8), dtype=torch.float16)
    value_current = torch.empty_like(key_current)
    workspace = make_gate0_seed_only_packed_workspace(q, seed_tokens=64)
    out = torch.empty_like(q)
    ring_index = torch.tensor([0], dtype=torch.int32)

    with pytest.raises(RuntimeError if not q.is_cuda else ValueError):
        gate0_seed_only_packed_ring_append_triton_forward_out(
            q,
            key_current,
            value_current,
            workspace["k_seed"],
            workspace["v_seed"],
            out,
            ring_index,
            block_size=32,
            sink_blocks=1,
            recent_blocks=0,
            middle_seed_blocks=1,
        )


def test_native_kv_cache_copies_prefill_and_appends():
    import torch

    policies = [
        Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0),
        Gate0SeedOnlyBatchedPolicy(policy_id="p2", model_id="m", layer_id=2),
    ]
    bundle = RouteBundle(
        policy_names=["p0", "p2"],
        policies=policies,
        artifacts=[{}, {}],
        layer_ids=[0, 2],
    )
    hf_cache = []
    for layer in range(3):
        k = torch.full((1, 2, 4, 3), float(layer))
        v = torch.full((1, 2, 4, 3), float(layer + 10))
        hf_cache.append((k, v))

    native = _native_cache_from_hf_cache(hf_cache, bundle, max_len=6)

    assert isinstance(native, StreamAttnNativeKVCache)
    assert native.summary()["shape"] == [2, 1, 2, 6, 3]
    assert torch.equal(native.k[0, :, :, :4, :], hf_cache[0][0])
    assert torch.equal(native.v[1, :, :, :4, :], hf_cache[2][1])

    k_new = torch.full((1, 2, 1, 3), 99.0)
    v_new = torch.full((1, 2, 1, 3), 199.0)
    k_layer, v_layer = native.append(2, k_new, v_new, torch.tensor([4]))

    assert torch.equal(k_layer[:, :, 4:5, :], k_new)
    assert torch.equal(v_layer[:, :, 4:5, :], v_new)

    k_layer, v_layer = native.append(0, k_new, v_new, 5)
    assert torch.equal(k_layer[:, :, 5:6, :], k_new)
    assert torch.equal(v_layer[:, :, 5:6, :], v_new)


def test_native_kv_cache_can_attach_hf_layer_views():
    import torch

    class Layer:
        def __init__(self, keys, values):
            self.keys = keys
            self.values = values

    class Cache:
        def __init__(self):
            self.layers = []

    policies = [Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=0)]
    bundle = RouteBundle(policy_names=["p0"], policies=policies, artifacts=[{}], layer_ids=[0])
    cache = Cache()
    cache.layers.append(
        Layer(
            torch.ones((1, 2, 4, 3)),
            torch.full((1, 2, 4, 3), 2.0),
        )
    )

    native = _native_cache_from_hf_cache(cache, bundle, max_len=6)

    assert cache.layers[0].keys.shape == (1, 2, 4, 3)
    assert cache.layers[0].keys.data_ptr() == native.k[0, :, :, :4, :].data_ptr()

    k_new = torch.full((1, 2, 1, 3), 9.0)
    v_new = torch.full((1, 2, 1, 3), 10.0)
    native.append(0, k_new, v_new, 4)

    assert cache.layers[0].keys.shape == (1, 2, 5, 3)
    assert cache.layers[0].keys.data_ptr() == native.k[0, :, :, :5, :].data_ptr()
    assert torch.equal(cache.layers[0].keys[:, :, 4:5, :], k_new)
    assert torch.equal(cache.layers[0].values[:, :, 4:5, :], v_new)


def test_parent_module_and_attr_resolves_nested_module():
    import torch

    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    root.model.layers[0].self_attn = torch.nn.Linear(2, 2)

    parent, attr = _parent_module_and_attr(root, "model.layers.0.self_attn")

    assert parent is root.model.layers[0]
    assert attr == "self_attn"


def test_streamattn_qwen_attention_module_delegates_patch():
    import torch

    class Original(torch.nn.Module):
        layer_idx = 3
        attention_type = "full_attention"

    class Patch:
        def __init__(self):
            self.called = False

        def forward(self, module, hidden_states, **kwargs):
            self.called = True
            assert module.layer_idx == 3
            return hidden_states + 1, None

    patch = Patch()
    wrapper = StreamAttnQwenAttentionModule(Original(), patch)
    hidden = torch.zeros((1, 1, 2))

    out, weights = wrapper(hidden, position_embeddings=(hidden, hidden), attention_mask=None)

    assert patch.called
    assert weights is None
    assert torch.equal(out, torch.ones_like(hidden))


def test_streamattn_qwen_attention_module_exposes_hot_projection_attrs():
    import torch

    class Original(torch.nn.Module):
        layer_idx = 3
        attention_type = "full_attention"
        head_dim = 4

        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8)
            self.k_proj = torch.nn.Linear(8, 4)
            self.v_proj = torch.nn.Linear(8, 4)
            self.o_proj = torch.nn.Linear(8, 8)

        def forward(self, *args, **kwargs):
            raise AssertionError("fallback should not run")

    patch = _SeedOnlyQwenDecodePatch(
        policy=Gate0SeedOnlyBatchedPolicy(policy_id="p0", model_id="m", layer_id=3),
        original_forward=lambda *args, **kwargs: None,
        packed_qkv_projection=True,
    )
    original = Original()
    wrapper = StreamAttnQwenAttentionModule(original, patch)

    assert wrapper.layer_idx == 3
    assert wrapper.attention_type == "full_attention"
    assert wrapper.head_dim == 4
    assert wrapper.q_proj is original.q_proj
    assert wrapper.o_proj is original.o_proj
    assert patch._packed_qkv_weight is not None
