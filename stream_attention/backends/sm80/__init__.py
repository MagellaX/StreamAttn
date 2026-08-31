"""Ampere (SM80) StreamAttn backends."""

from .paged_gqa_exact import compile_sm80_paged_gqa_extension
from .tk_grouped_exact import (
    ExactDecodePlan,
    PROMOTED_EXACT_G4_SPLITS,
    PROMOTED_EXACT_G8_SPLITS,
    PROMOTED_EXACT_SHAPES,
    compile_grouped_exact_extension,
    resolve_tk_root,
    supports_grouped_exact,
)

__all__ = [
    "ExactDecodePlan",
    "PROMOTED_EXACT_G4_SPLITS",
    "PROMOTED_EXACT_G8_SPLITS",
    "PROMOTED_EXACT_SHAPES",
    "compile_grouped_exact_extension",
    "compile_sm80_paged_gqa_extension",
    "resolve_tk_root",
    "supports_grouped_exact",
]
