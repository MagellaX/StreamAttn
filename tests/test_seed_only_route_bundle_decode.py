from benchmarks.profile_gate0_seed_only_multi_layer_rollout import RouteBundle
from benchmarks.profile_seed_only_route_bundle_decode import (
    StreamAttnNativeKVCache,
    _native_cache_from_hf_cache,
    apply_layer_seed_overrides,
    parse_layer_id_set,
    parse_layer_seed_overrides,
    summarize_patch_timing_rows,
)
from stream_attention.decode import Gate0SeedOnlyBatchedPolicy


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


def test_parse_layer_id_set_accepts_commas_and_semicolons():
    assert parse_layer_id_set("") == set()
    assert parse_layer_id_set("0,14;16") == {0, 14, 16}


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
