"""Ampere (SM80) StreamAttn backends."""

from .paged_gqa_exact import compile_sm80_paged_gqa_extension

__all__ = ["compile_sm80_paged_gqa_extension"]
