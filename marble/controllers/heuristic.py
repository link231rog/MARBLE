from __future__ import annotations

import re
from typing import Optional, Sequence

from marble.memory.schema import MemoryItem, MemoryProposal, MemoryTargetState


_SHARED_SIGNALS = re.compile(
    r"\b(share|shared|all|team|decision|result|consensus|plan|finding)\b",
    re.IGNORECASE,
)


def _find_supersedes(
    proposal: MemoryProposal,
    current_state: Sequence[MemoryItem],
) -> Optional[str]:
    normalized = proposal.title.strip().casefold()
    for item in current_state:
        if (
            item.active
            and item.task_id == proposal.task_id
            and item.title.strip().casefold() == normalized
        ):
            return item.memory_id
    return None


class GlobalAlwaysController:
    """Admit every proposal as global memory."""

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        return MemoryTargetState(
            exists=True,
            visibility="global",
            owner_id=None,
            supersedes=_find_supersedes(proposal, current_state),
        )


class AbsentController:
    """Reject every proposal."""

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        return MemoryTargetState(exists=False, visibility="absent")


class PrivateOnlyController:
    """Store every proposal as private to its author."""

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        return MemoryTargetState(
            exists=True,
            visibility="private",
            owner_id=proposal.agent_id,
            supersedes=_find_supersedes(proposal, current_state),
        )


class LTSStyleController:
    """Binary sharing baseline: reject local notes, publish shared findings.

    This is an explicit ``absent/global`` policy and never emits private
    memories.  It is kept separate from the proposed three-way controller.
    """

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        if not proposal.raw_value.strip() or not _SHARED_SIGNALS.search(proposal.title):
            return MemoryTargetState(False, "absent")
        return MemoryTargetState(
            True,
            "global",
            supersedes=_find_supersedes(proposal, current_state),
        )


class HeuristicController:
    """Private by default, global on shared-signal titles, absent when empty."""

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        supersedes = _find_supersedes(proposal, current_state)
        if not proposal.raw_value.strip():
            return MemoryTargetState(False, "absent")
        if _SHARED_SIGNALS.search(proposal.title):
            return MemoryTargetState(True, "global", supersedes=supersedes)
        return MemoryTargetState(
            True,
            "private",
            owner_id=proposal.agent_id,
            supersedes=supersedes,
        )
