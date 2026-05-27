from benchmarks.profile_gate0_seed_only_multi_layer_rollout import RouteBundle
from benchmarks.profile_seed_only_route_bundle_decode import (
    apply_layer_seed_overrides,
    parse_layer_seed_overrides,
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
