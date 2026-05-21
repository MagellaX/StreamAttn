from benchmarks.profile_gate0_kv_group_repair_real import (
    corrected_max_error,
    parse_budgets,
    q_heads_for_kv_group,
    repair_work_summary,
    select_repair_heads,
)


def test_q_heads_for_true_gqa_kv_group():
    assert q_heads_for_kv_group(0, q_heads=14, kv_heads=2) == [0, 1, 2, 3, 4, 5, 6]
    assert q_heads_for_kv_group(1, q_heads=14, kv_heads=2) == [7, 8, 9, 10, 11, 12, 13]


def test_budget_repair_selection_and_corrected_error():
    rows = [
        {"head": 2, "max_abs_error": 0.020},
        {"head": 3, "max_abs_error": 0.004},
        {"head": 4, "max_abs_error": 0.011},
    ]

    budgets = parse_budgets("strict:1e-2,moderate:1.5e-2")

    strict_repair = select_repair_heads(rows, budget=budgets[0]["max_abs_error"])
    moderate_repair = select_repair_heads(rows, budget=budgets[1]["max_abs_error"])

    assert strict_repair == [2, 4]
    assert corrected_max_error(rows, repair_heads=strict_repair) == 0.004
    assert moderate_repair == [2]
    assert corrected_max_error(rows, repair_heads=moderate_repair) == 0.011


def test_repair_work_summary_marks_same_kernel_repair_problem():
    summary = repair_work_summary(
        group_size=7,
        repair_count=1,
        kv_len=32768,
        block_size=32,
        sink_blocks=2,
        recent_blocks=2,
        middle_seed_blocks=8,
    )

    assert summary["num_blocks"] == 1024
    assert summary["seed_blocks"] == 12
    assert summary["seed_fraction"] == 12 / 1024
    assert summary["whole_group_seed_only_can_skip_nonseed_kv"] is True
    assert summary["same_kernel_repair_forces_nonseed_kv_when_repair_count_positive"] is True
    assert summary["external_seed_plus_repair_row_block_fraction"] < 0.2
