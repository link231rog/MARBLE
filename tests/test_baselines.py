import pytest
from marble.experiments.baselines import (
    MAIN_BASELINES,
    baseline_spec,
    canonical_baseline,
)


def test_baselines_spec_and_resolution():
    # Frozen experiment matrix
    assert MAIN_BASELINES == (
        "single_agent",
        "no_memory",
        "global_add_all",
        "lts_style",
        "mem0_style",
        "memory_r1_style",
        "g_memory_style",
        "collabmem_style",
        "copper_style",
        "ours_base",
        "ours_sft",
        "ours_rl",
    )

    # Legacy names resolve to canonical methods
    for legacy, canonical in [
        ("global_always", "global_add_all"),
        ("qwen_sft", "ours_sft"),
        ("qwen_rl", "ours_rl"),
        ("ours-private-to-global", "ours_private_to_global"),
    ]:
        assert canonical_baseline(legacy) == canonical

    # Baseline specs
    assert baseline_spec("single_agent").uses_memory is False
    assert baseline_spec("single_agent").single_agent is True
    assert baseline_spec("memory_r1_style").requires_crud_manager is True
    assert baseline_spec("memory_r1_style").controller == "memory_r1_crud"

    with pytest.raises(ValueError, match="unknown baseline"):
        canonical_baseline("not-a-method")

    for name in ("ours_base", "ours_sft", "ours_rl", "single_agent", "no_memory", "global_add_all", "lts_style", "mem0_style"):
        assert baseline_spec(name).enable_comm_governor is False
