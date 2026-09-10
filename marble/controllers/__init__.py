from __future__ import annotations

from typing import Protocol, Sequence

from marble.memory.schema import MemoryItem, MemoryProposal, MemoryTargetState

from .heuristic import (
    AbsentController,
    GlobalAlwaysController,
    HeuristicController,
    LTSStyleController,
    PrivateOnlyController,
)
from .json_controller import VALID_VISIBILITIES, JsonController
from .local_policy import LocalPolicyController, features


class MemoryController(Protocol):
    """Outputs a target state; the bank applies the diff."""

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState: ...

__all__ = [
    "AbsentController",
    "GlobalAlwaysController",
    "HeuristicController",
    "JsonController",
    "LTSStyleController",
    "LocalPolicyController",
    "MemoryController",
    "PrivateOnlyController",
    "VALID_VISIBILITIES",
    "features",
]
