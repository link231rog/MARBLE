from __future__ import annotations

from threading import RLock
from typing import Dict, List, Optional

from .schema import (
    MemoryCard,
    MemoryItem,
    MemoryProposal,
    MemoryTargetState,
)


class MemoryBank:
    """Task-scoped memory store with explicit visibility checks."""

    def __init__(self) -> None:
        self._items: Dict[str, MemoryItem] = {}
        self._proposal_to_memory: Dict[str, str] = {}
        self._clock = 0
        self._lock = RLock()

    def apply(
        self,
        proposal: MemoryProposal,
        target: MemoryTargetState,
    ) -> Optional[MemoryItem]:
        if proposal.task_id == "":
            raise ValueError("proposal task_id must not be empty")
        if target.visibility == "private" and target.owner_id != proposal.agent_id:
            raise ValueError("private memory owner must equal proposal agent_id")
        if target.supersedes is not None:
            self._validate_supersession(proposal, target.supersedes)
        if not target.exists:
            return None

        with self._lock:
            self._clock += 1
            memory_id = proposal.proposal_id
            if memory_id in self._items:
                raise ValueError("proposal_id has already been applied")
            item = MemoryItem(
                memory_id=memory_id,
                proposal_id=proposal.proposal_id,
                task_id=proposal.task_id,
                title=proposal.title,
                raw_value=proposal.raw_value,
                visibility=target.visibility,
                owner_id=target.owner_id,
                source_agent=proposal.agent_id,
                source=proposal.source,
                step_index=proposal.step_index,
                active=True,
                supersedes=target.supersedes,
                created_at=self._clock,
            )
            self._items[memory_id] = item
            self._proposal_to_memory[proposal.proposal_id] = memory_id
            if target.supersedes is not None:
                old_item = self._items[target.supersedes]
                self._items[target.supersedes] = MemoryItem(
                    memory_id=old_item.memory_id,
                    proposal_id=old_item.proposal_id,
                    task_id=old_item.task_id,
                    title=old_item.title,
                    raw_value=old_item.raw_value,
                    visibility=old_item.visibility,
                    owner_id=old_item.owner_id,
                    source_agent=old_item.source_agent,
                    source=old_item.source,
                    step_index=old_item.step_index,
                    active=False,
                    supersedes=old_item.supersedes,
                    created_at=old_item.created_at,
                )
            return item

    def visible_keys(self, reader_id: str, task_id: str) -> List[MemoryCard]:
        with self._lock:
            return [
                item.card()
                for item in self._items.values()
                if item.active
                and item.task_id == task_id
                and (
                    item.visibility == "global"
                    or item.owner_id == reader_id
                )
            ]

    def read(self, memory_id: str, reader_id: str, task_id: str) -> MemoryItem:
        with self._lock:
            item = self._items[memory_id]
            if not item.active or item.task_id != task_id:
                raise KeyError(memory_id)
            if item.visibility == "private" and item.owner_id != reader_id:
                raise PermissionError(memory_id)
            return item

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        with self._lock:
            return self._items.get(memory_id)

    def all_items(self) -> List[MemoryItem]:
        with self._lock:
            return list(self._items.values())

    def _validate_supersession(
        self,
        proposal: MemoryProposal,
        memory_id: str,
    ) -> None:
        with self._lock:
            old_item = self._items.get(memory_id)
            if old_item is None:
                raise KeyError(memory_id)
            if not old_item.active:
                raise ValueError("cannot supersede inactive memory")
            if old_item.task_id != proposal.task_id:
                raise ValueError("memory task_id does not match proposal task_id")
