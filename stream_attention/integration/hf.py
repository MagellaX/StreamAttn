import logging
from typing import Optional
import torch.nn as nn

try:
    from transformers import PreTrainedModel
    from transformers.models.llama.modeling_llama import LlamaAttention
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

    export_hf = True
except Exception:  # transformers may not be installed
    export_hf = False
    PreTrainedModel = object  # type: ignore
    LlamaAttention = object  # type: ignore
    GPT2Attention = object  # type: ignore

from stream_attention.core.attention import StreamAttention
from stream_attention.core.config import StreamAttentionConfig

logger = logging.getLogger(__name__)


def replace_llama_attention(
    model, config: StreamAttentionConfig, module_name: str = "self_attn"
) -> int:
    """
    Replace LlamaAttention modules with StreamAttention.
    Returns number of modules replaced.
    """
    if not export_hf:
        logger.warning("transformers not available; cannot replace LlamaAttention")
        return 0
    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, LlamaAttention) and name.endswith(module_name):
            parent_name = ".".join(name.split(".")[:-1])
            attr = name.split(".")[-1]
            parent = model
            for part in parent_name.split(".") if parent_name else []:
                parent = getattr(parent, part)
            try:
                param = next(module.parameters())
                device = param.device
                dtype = param.dtype
            except StopIteration:
                device = None
                dtype = None
            new_mod = StreamAttention(config)
            if device is not None:
                new_mod = new_mod.to(device=device, dtype=dtype)
            setattr(parent, attr, new_mod)
            replaced += 1
            logger.info(f"Replaced {name} with StreamAttention (aligned to device/dtype)")
    return replaced


def replace_attention_generic(
    model, config: StreamAttentionConfig, name_pattern: str = "attn"
) -> int:
    """Heuristically replace modules whose name contains pattern with StreamAttention."""
    replaced = 0
    for name, module in model.named_modules():
        if name_pattern in name:
            parent_name = ".".join(name.split(".")[:-1])
            attr = name.split(".")[-1]
            parent = model
            for part in parent_name.split(".") if parent_name else []:
                parent = getattr(parent, part)
            try:
                param = next(module.parameters())
                device = param.device
                dtype = param.dtype
            except Exception:
                device = None
                dtype = None
            try:
                new_mod = StreamAttention(config)
                if device is not None:
                    new_mod = new_mod.to(device=device, dtype=dtype)
                setattr(parent, attr, new_mod)
                replaced += 1
            except Exception:
                continue
    return replaced


class GPT2AttentionAdapter(nn.Module):
    """
    Adapter that wraps StreamAttention to match GPT2Attention forward signature.
    It projects hidden_states to QKV internally and exposes GPT-2-compatible kwargs.
    """

    def __init__(self, config: StreamAttentionConfig):
        super().__init__()
        from stream_attention.core.attention import StreamAttention

        # Ensure projections for hidden_states -> QKV
        cfg = StreamAttentionConfig(**{**config.__dict__, "use_qkv_projections": True})
        self.inner = StreamAttention(cfg)

    def forward(
        self,
        hidden_states=None,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        past_kv = None
        if (
            layer_past is not None
            and isinstance(layer_past, tuple)
            and len(layer_past) == 2
        ):
            past_kv = (layer_past[0], layer_past[1])
        output = self.inner(
            hidden_states=hidden_states,
            past_key_value=past_kv,
            use_cache=use_cache,
            causal=True,
        )
        if use_cache:
            attn_out, kv = output
            if output_attentions:
                return attn_out, None, kv
            return attn_out, kv
        else:
            if output_attentions:
                return output, None
            return output


def replace_gpt2_attention(
    model, config: StreamAttentionConfig, module_name: str = "attn"
) -> int:
    """Replace GPT2Attention modules with a GPT2AttentionAdapter wrapping StreamAttention."""
    if not export_hf:
        logger.warning("transformers not available; cannot replace GPT2Attention")
        return 0
    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, GPT2Attention) and name.endswith(module_name):
            parent_name = ".".join(name.split(".")[:-1])
            attr = name.split(".")[-1]
            parent = model
            for part in parent_name.split(".") if parent_name else []:
                parent = getattr(parent, part)
            try:
                param = next(module.parameters())
                device = param.device
                dtype = param.dtype
            except StopIteration:
                device = None
                dtype = None
            adapter = GPT2AttentionAdapter(config)
            if device is not None:
                adapter = adapter.to(device=device, dtype=dtype)
            setattr(parent, attr, adapter)
            replaced += 1
            logger.info(f"Replaced {name} with GPT2AttentionAdapter(StreamAttention) aligned to device/dtype")
    return replaced
