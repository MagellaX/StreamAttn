from benchmarks.profile_gate0_seed_only_multi_layer_rollout import RouteBundle
from benchmarks.profile_seed_only_route_bundle_decode import (
    StreamAttnNativeKVCache,
    StreamAttnQwenAttentionModule,
    _SeedOnlyQwenDecodePatch,
    _native_cache_from_hf_cache,
    _native_cache_mask_bookkeeping,
    _parent_module_and_attr,
    apply_layer_seed_overrides,
    parse_layer_id_set,
    parse_layer_seed_overrides,
    summarize_patch_timing_rows,
)
from stream_attention.decode import Gate0SeedOnlyBatchedPolicy
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


def test_parse_layer_id_set_accepts_commas_and_semicolons():
    assert parse_layer_id_set("") == set()
    assert parse_layer_id_set("0,14;16") == {0, 14, 16}


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
