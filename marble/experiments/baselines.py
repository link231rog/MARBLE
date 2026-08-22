from __future__ import annotations

from typing import Optional

from marble.controllers import (
    AbsentController,
    GlobalAlwaysController,
    HeuristicController,
    PrivateOnlyController,
)
from marble.memory import GovernedMemory, MemoryBank, TraceLogger

BASELINES = ("no_memory", "global_always", "private_only", "heuristic")

_CONTROLLERS = {
    "no_memory": AbsentController,
    "global_always": GlobalAlwaysController,
    "private_only": PrivateOnlyController,
    "heuristic": HeuristicController,
}


def make_memory(
    baseline: str,
    trace_path: Optional[str] = None,
) -> GovernedMemory:
    if baseline not in _CONTROLLERS:
        raise ValueError(f"unknown baseline: {baseline}")
    return GovernedMemory(
        bank=MemoryBank(),
        controller=_CONTROLLERS[baseline](),
        trace=TraceLogger(trace_path) if trace_path else None,
    )
