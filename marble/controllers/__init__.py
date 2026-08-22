from __future__ import annotations

from typing import Protocol, Sequence

from marble.memory.schema import MemoryItem, MemoryProposal, MemoryTargetState


class MemoryController(Protocol):
    """Outputs a target state; the bank applies the diff."""

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState: ...


from .heuristic import (
    AbsentController,
    GlobalAlwaysController,
    HeuristicController,
    PrivateOnlyController,
)

__all__ = [
    "AbsentController",
    "GlobalAlwaysController",
    "HeuristicController",
    "MemoryController",
    "PrivateOnlyController",
]
