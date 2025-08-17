import logging
from typing import Optional

try:
	from transformers import PreTrainedModel
	from transformers.models.llama.modeling_llama import LlamaAttention
	export_hf = True
except ImportError:  # transformers may not be installed
	export_hf = False
	PreTrainedModel = object  # type: ignore
	LlamaAttention = object  # type: ignore

from stream_attention.core.attention import StreamAttention
from stream_attention.core.config import StreamAttentionConfig

logger = logging.getLogger(__name__)


def replace_llama_attention(model: PreTrainedModel, config: StreamAttentionConfig, module_name: str = "self_attn") -> int:
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
			parent_name = '.'.join(name.split('.')[:-1])
			attr = name.split('.')[-1]
			parent = model
			for part in parent_name.split('.') if parent_name else []:
				parent = getattr(parent, part)
			setattr(parent, attr, StreamAttention(config))
			replaced += 1
			logger.info(f"Replaced {name} with StreamAttention")
	return replaced


def replace_attention_generic(model: PreTrainedModel, config: StreamAttentionConfig, name_pattern: str = "attn") -> int:
	"""Heuristically replace modules whose name contains pattern with StreamAttention."""
	replaced = 0
	for name, module in model.named_modules():
		if name_pattern in name:
			parent_name = '.'.join(name.split('.')[:-1])
			attr = name.split('.')[-1]
			parent = model
			for part in parent_name.split('.') if parent_name else []:
				parent = getattr(parent, part)
			try:
				setattr(parent, attr, StreamAttention(config))
				replaced += 1
			except Exception:
				continue
	return replaced