"""Canonical experiment methods and their runtime policies.

The registry is deliberately declarative.  It keeps paper-facing method names
stable while the runner maps each method to an implementation component.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from marble.controllers import (
    AbsentController,
    GlobalAlwaysController,
    HeuristicController,
    LTSStyleController,
    PrivateOnlyController,
)
from marble.memory import GovernedMemory, MemoryBank, TraceLogger


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    controller: str
    uses_memory: bool
    single_agent: bool = False
    requires_crud_manager: bool = False


MAIN_BASELINES: Tuple[str, ...] = (
    "single_agent",
    "no_memory",
    "global_add_all",
    "lts_style",
    "memory_r1_style",
    "ours_sft",
    "ours_rl",
)

# Legacy offline coding-rollout methods.  The real benchmark runner imports
# MAIN_BASELINES directly, while this compatibility surface keeps pre-existing
# fast tests and examples independent of Qwen/CRUD runtime dependencies.
BASELINES: Tuple[str, ...] = (
    "no_memory",
    "global_always",
    "private_only",
    "heuristic",
)

BASELINE_REGISTRY: Dict[str, BaselineSpec] = {
    "single_agent": BaselineSpec(
        "single_agent", controller="absent", uses_memory=False, single_agent=True
    ),
    "no_memory": BaselineSpec("no_memory", controller="absent", uses_memory=False),
    "global_add_all": BaselineSpec(
        "global_add_all", controller="global_add_all", uses_memory=True
    ),
    "lts_style": BaselineSpec("lts_style", controller="lts_binary", uses_memory=True),
    "memory_r1_style": BaselineSpec(
        "memory_r1_style",
        controller="memory_r1_crud",
        uses_memory=True,
        requires_crud_manager=True,
    ),
    "ours_base": BaselineSpec("ours_base", controller="qwen_sft", uses_memory=True),
    "ours_sft": BaselineSpec("ours_sft", controller="qwen_sft", uses_memory=True),
    "ours_rl": BaselineSpec("ours_rl", controller="qwen_rl", uses_memory=True),
    # Auxiliary diagnostics are intentionally excluded from MAIN_BASELINES.
    "private_only": BaselineSpec("private_only", controller="private_only", uses_memory=True),
    "heuristic": BaselineSpec("heuristic", controller="heuristic", uses_memory=True),
    "learned_controller": BaselineSpec(
        "learned_controller", controller="local_policy", uses_memory=True
    ),
}

ALIASES = {
    "global_always": "global_add_all",
    "qwen_base": "ours_base",
    "ours_prompted": "ours_base",
    "qwen_sft": "ours_sft",
    "qwen_rl": "ours_rl",
}


def canonical_baseline(name: str) -> str:
    """Resolve a legacy spelling, raising a useful error for unknown methods."""
    canonical = ALIASES.get(name, name)
    if canonical not in BASELINE_REGISTRY:
        choices = ", ".join((*MAIN_BASELINES, "private_only", "heuristic", "learned_controller"))
        raise ValueError(f"unknown baseline {name!r}; choose from {choices} or 'multi'")
    return canonical


def baseline_spec(name: str) -> BaselineSpec:
    return BASELINE_REGISTRY[canonical_baseline(name)]


def make_memory(
    baseline: str,
    trace_path: Optional[str] = None,
) -> GovernedMemory:
    """Create the lightweight governed-memory fixture used by offline rollouts."""
    spec = baseline_spec(baseline)
    controllers = {
        "absent": AbsentController,
        "global_add_all": GlobalAlwaysController,
        "lts_binary": LTSStyleController,
        "private_only": PrivateOnlyController,
        "heuristic": HeuristicController,
    }
    controller_cls = controllers.get(spec.controller)
    if controller_cls is None:
        raise ValueError(
            f"{baseline!r} needs the full MultiAgentBench runner, not make_memory"
        )
    return GovernedMemory(
        bank=MemoryBank(),
        controller=controller_cls(),
        trace=TraceLogger(trace_path) if trace_path else None,
    )
