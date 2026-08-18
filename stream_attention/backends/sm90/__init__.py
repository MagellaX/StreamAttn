"""Hopper (SM90) StreamAttn backends."""

from .transposed_gqa_exact import (
    ExactDecodePlan,
    PROMOTED_EXACT_D128_G4_SPLITS,
    PROMOTED_EXACT_G4_SPLITS,
    PROMOTED_EXACT_SHAPE,
    PROMOTED_EXACT_SHAPES,
    PROMOTED_EXACT_SPLITS,
    choose_num_splits,
    compile_transposed_gqa_exact_extension,
    supports_transposed_gqa_exact,
)

__all__ = [
    "ExactDecodePlan",
    "PROMOTED_EXACT_D128_G4_SPLITS",
    "PROMOTED_EXACT_G4_SPLITS",
    "PROMOTED_EXACT_SHAPE",
    "PROMOTED_EXACT_SHAPES",
    "PROMOTED_EXACT_SPLITS",
    "choose_num_splits",
    "compile_transposed_gqa_exact_extension",
    "supports_transposed_gqa_exact",
]
