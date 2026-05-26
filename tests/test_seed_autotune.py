import math

from stream_attention import (
    SeedKernelShape,
    autotune_seed_kernel_mode,
    load_packaged_gate0_seed_only_batched_policy,
    seed_shape_from_policy,
)


def test_qwen_l8_seed_shape_locks_duplication_math():
    policy = load_packaged_gate0_seed_only_batched_policy()
    shape = seed_shape_from_policy(policy, batch=8)

    assert shape.group_size == 7
    assert shape.seed_tokens == 384
    assert math.isclose(shape.seed_token_ratio, 384.0 / 32768.0)
    assert math.isclose(shape.head_private_kv_byte_ratio, 7.0 * 384.0 / 32768.0)
    assert shape.head_private_kv_byte_ratio < 0.10
    assert shape.head_private_seed_kv_bytes < shape.exact_kv_bytes


def test_qwen_l8_batch8_prefers_head_private_direct():
    policy = load_packaged_gate0_seed_only_batched_policy()
    shape = seed_shape_from_policy(policy, batch=8)

    result = autotune_seed_kernel_mode(
        shape,
        sm_count=132,
        target_waves=0.75,
        duplication_byte_budget=0.15,
    )

    assert result.decision == "seed_only_native_candidate"
    assert result.recommended_mode == "head_private_direct_seed"
    assert result.occupancy_threshold_ctas == 99
    best = result.candidates[0]
    assert best.mode == "head_private_direct_seed"
    assert best.cta_count == 8 * 14
    assert best.viable


def test_qwen_l8_batch4_prefers_split_seed_to_raise_ctas():
    policy = load_packaged_gate0_seed_only_batched_policy()
    shape = seed_shape_from_policy(policy, batch=4)

    result = autotune_seed_kernel_mode(
        shape,
        sm_count=132,
        target_waves=0.75,
        seed_tile_tokens=(384, 192, 128, 64, 32),
        duplication_byte_budget=0.15,
    )

    assert result.decision == "seed_only_native_candidate"
    assert result.recommended_mode == "head_private_split_seed"
    assert result.recommended_seed_tile_tokens in {192, 128, 64, 32}
    direct = next(
        candidate for candidate in result.candidates if candidate.mode == "head_private_direct_seed"
    )
    assert direct.cta_count == 4 * 14
    assert not direct.viable_occupancy
    assert result.candidates[0].cta_count >= result.occupancy_threshold_ctas


def test_seed_autotune_rejects_excessive_duplication():
    shape = SeedKernelShape(
        batch=8,
        q_heads=14,
        kv_heads=2,
        kv_len=2048,
        dim=64,
        block_size=32,
        sink_blocks=2,
        recent_blocks=2,
        middle_seed_blocks=8,
    )

    result = autotune_seed_kernel_mode(
        shape,
        sm_count=132,
        target_waves=0.75,
        duplication_byte_budget=0.15,
    )

    assert shape.head_private_kv_byte_ratio > 1.0
    assert result.recommended_mode != "head_private_direct_seed"
    assert result.decision in {
        "seed_only_native_candidate",
        "exact_native_until_kernel_or_policy_changes",
    }
