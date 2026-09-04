"""Hopper (SM90) StreamAttn backends."""

from .grouped_gqa_prefill import (
    GroupedRSPrefillPlan,
    GroupedWgmmaPrefillPlan,
    decode_grouped_prefill_resources,
    decode_grouped_rs_prefill_resources,
    supports_grouped_rs_prefill,
    supports_grouped_wgmma_prefill,
)
from .micro_prefill import (
    MICRO_PREFILL_MAX_QUERY_LEN,
    MICRO_PREFILL_MIN_QUERY_LEN,
    MICRO_PREFILL_QUERY_TILE_ROWS,
    MICRO_PREFILL_SUPPORTED_GROUPS,
    MicroPrefillPlan,
    NaturalMicroPrefillPlan,
    choose_micro_prefill_splits,
    choose_natural_micro_prefill_splits,
    micro_prefill_shape_reasons,
    natural_micro_prefill_query_tiles,
    supports_sm90_micro_prefill,
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
    "MICRO_PREFILL_MAX_QUERY_LEN",
    "MICRO_PREFILL_MIN_QUERY_LEN",
    "MICRO_PREFILL_QUERY_TILE_ROWS",
    "MICRO_PREFILL_SUPPORTED_GROUPS",
    "MicroPrefillPlan",
    "NaturalMicroPrefillPlan",
    "PROMOTED_EXACT_D128_G4_SPLITS",
    "PROMOTED_EXACT_G4_SPLITS",
    "PROMOTED_EXACT_SHAPE",
    "PROMOTED_EXACT_SHAPES",
    "PROMOTED_EXACT_SPLITS",
    "choose_num_splits",
    "choose_micro_prefill_splits",
    "choose_natural_micro_prefill_splits",
    "compile_transposed_gqa_exact_extension",
    "decode_grouped_prefill_resources",
    "decode_grouped_rs_prefill_resources",
    "micro_prefill_shape_reasons",
    "natural_micro_prefill_query_tiles",
    "supports_grouped_rs_prefill",
    "supports_grouped_wgmma_prefill",
    "supports_sm90_micro_prefill",
    "supports_transposed_gqa_exact",
]
