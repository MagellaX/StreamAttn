"""Hopper (SM90) StreamAttn backends."""

from .grouped_gqa_prefill import (
    GroupedRSPrefillPlan,
    GroupedWgmmaPrefillPlan,
    decode_grouped_prefill_resources,
    decode_grouped_rs_prefill_resources,
    supports_grouped_rs_prefill,
    supports_grouped_wgmma_prefill,
)

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
    "GroupedRSPrefillPlan",
    "GroupedWgmmaPrefillPlan",
    "PROMOTED_EXACT_D128_G4_SPLITS",
    "PROMOTED_EXACT_G4_SPLITS",
    "PROMOTED_EXACT_SHAPE",
    "PROMOTED_EXACT_SHAPES",
    "PROMOTED_EXACT_SPLITS",
    "choose_num_splits",
    "compile_transposed_gqa_exact_extension",
    "decode_grouped_prefill_resources",
    "decode_grouped_rs_prefill_resources",
    "supports_grouped_rs_prefill",
    "supports_grouped_wgmma_prefill",
    "supports_transposed_gqa_exact",
]
