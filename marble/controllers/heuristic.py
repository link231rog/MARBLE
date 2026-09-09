from __future__ import annotations

import re
from typing import Callable, Optional, Sequence

from marble.memory.schema import MemoryItem, MemoryProposal, MemoryTargetState


_SHARED_SIGNALS = re.compile(
    r"\b(share|shared|all|team|decision|result|consensus|plan|finding|query_db|tool|"
    r"observation|identified|bottleneck|workload|analysis|summary|literature|paper|"
    r"detected|index|select|insert|vacuum|lock|stat|evidence|privacy|method|model|"
    r"learning|gap|hypothesis|recommend|evaluation)\b",
    re.IGNORECASE,
)


_NON_SUPERSEDABLE_TITLES = {
    "", "empty observation", "<concise summary key>", "concise summary key",
    "summary key", "we need to produce final answer", "we need to produce the next task",
    "we need to produce the answer", "we need to produce 5q", "we need to produce a 5q research proposal",
    "we have to produce final answer after considering all root causes",
    "we need to respond as agent2", "we need to respond as agent3",
}


def _find_update_target(
    proposal: MemoryProposal,
    current_state: Sequence[MemoryItem],
) -> Optional[str]:
    normalized = proposal.title.strip().casefold()
    if not normalized or normalized in _NON_SUPERSEDABLE_TITLES or (normalized.startswith("<") and normalized.endswith(">")):
        return None
    for item in current_state:
        if (
            item.active
            and item.task_id == proposal.task_id
            and (item.visibility == "global" or proposal.agent_id in getattr(item, "target_recipients", ()))
            and item.title.strip().casefold() == normalized
        ):
            return item.memory_id
    return None

_find_supersedes = _find_update_target


class GlobalAlwaysController:
    """Admit every proposal as global memory."""

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        up = _find_update_target(proposal, current_state)
        return MemoryTargetState(
            exists=True,
            visibility="global",
            target_recipients=(),
            supersedes=up,
            update=up,
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
    """Store every proposal as private to its author (routed to author only)."""

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        up = _find_update_target(proposal, current_state)
        return MemoryTargetState(
            exists=True,
            visibility="targeted",
            target_recipients=(proposal.agent_id,),
            supersedes=up,
            update=up,
        )


class LTSStyleController:
    """Binary sharing baseline: reject local notes, publish shared findings.

    This is an explicit ``absent/global`` policy following LTS (ICML 2026).
    It never emits private memories. Supports an optional LLM admission judge (LTS-LLM)
    or deterministic signal rules as fast fallback.
    """

    def __init__(
        self,
        judge_fn: Optional[Callable[[MemoryProposal], bool]] = None,
    ) -> None:
        self.judge_fn = judge_fn

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        title = proposal.title.strip().lower()
        if not proposal.raw_value.strip() or not title:
            return MemoryTargetState(False, "absent")
        if title in {"empty observation", "none", "noop"} or "local" in title or "scratch" in title:
            return MemoryTargetState(False, "absent")

        up = _find_update_target(proposal, current_state)
        if self.judge_fn is not None:
            try:
                admitted = self.judge_fn(proposal)
                if not admitted:
                    return MemoryTargetState(False, "absent")
                return MemoryTargetState(
                    True,
                    "global",
                    supersedes=up,
                    update=up,
                )
            except Exception:
                pass  # Fall back to signal match

        if not _SHARED_SIGNALS.search(proposal.title):
            return MemoryTargetState(False, "absent")
        return MemoryTargetState(
            True,
            "global",
            supersedes=up,
            update=up,
        )


class HeuristicController:
    """Targeted to author by default, global on shared-signal titles, absent when empty."""

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        up = _find_update_target(proposal, current_state)
        if not proposal.raw_value.strip():
            return MemoryTargetState(False, "absent")
        if _SHARED_SIGNALS.search(proposal.title):
            return MemoryTargetState(True, "global", target_recipients=(), supersedes=up, update=up)
        return MemoryTargetState(
            True,
            "targeted",
            target_recipients=(proposal.agent_id,),
            supersedes=up,
            update=up,
        )
