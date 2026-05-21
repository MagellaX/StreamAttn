from benchmarks.profile_thunderkittens_extension_smoke import _tk_arch_define


def test_tk_arch_define_hopper() -> None:
    assert _tk_arch_define("sm_90a") == "KITTENS_SM90"
    assert _tk_arch_define("compute_90a") == "KITTENS_SM90"


def test_tk_arch_define_blackwell_variants() -> None:
    assert _tk_arch_define("sm_100a") == "KITTENS_SM100"
    assert _tk_arch_define("sm_103a") == "KITTENS_SM103"
    assert _tk_arch_define("sm_120a") == "KITTENS_SM120"
