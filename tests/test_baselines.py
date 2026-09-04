import pytest

from marble.experiments.baselines import (
    MAIN_BASELINES,
    baseline_spec,
    canonical_baseline,
)


def test_main_baselines_match_frozen_experiment_matrix():
    assert MAIN_BASELINES == (
        "single_agent",
        "no_memory",
        "global_add_all",
        "lts_style",
        "mem0_style",
        "amem_style",
        "memoryos_style",
        "memory_r1_style",
        "g_memory_style",
        "collabmem_style",
        "copper_style",
        "ours_base",
        "ours_sft",
        "ours_rl",
    )



@pytest.mark.parametrize(
    ("legacy", "canonical"),
    [
        ("global_always", "global_add_all"),
        ("qwen_sft", "ours_sft"),
        ("qwen_rl", "ours_rl"),
    ],
)
def test_legacy_names_resolve_to_canonical_methods(legacy, canonical):
    assert canonical_baseline(legacy) == canonical


def test_single_agent_spec_has_no_memory_and_single_agent_execution():
    spec = baseline_spec("single_agent")

    assert spec.uses_memory is False
    assert spec.single_agent is True


def test_memory_r1_spec_is_explicitly_distinct_from_ours():
    spec = baseline_spec("memory_r1_style")

    assert spec.requires_crud_manager is True
    assert spec.controller == "memory_r1_crud"


def test_unknown_baseline_is_rejected():
    with pytest.raises(ValueError, match="unknown baseline"):
        canonical_baseline("not-a-method")
