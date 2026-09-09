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
        update_target = target.update if target.update is not None else target.supersedes
        if update_target is not None:
            self._validate_update(proposal, update_target)
        if not target.exists:
            return None

        with self._lock:
            self._clock += 1
            memory_id = proposal.proposal_id
            if memory_id in self._items:
                raise ValueError("proposal_id has already been applied")
            target_recipients = tuple(getattr(target, "target_recipients", ()))
            item = MemoryItem(
                memory_id=memory_id,
                proposal_id=proposal.proposal_id,
                task_id=proposal.task_id,
                title=proposal.title,
                raw_value=proposal.raw_value,
                visibility=target.visibility,
                source_agent=proposal.agent_id,
                source=proposal.source,
                step_index=proposal.step_index,
                active=True,
                supersedes=update_target,
                update=update_target,
                created_at=self._clock,
                summary=proposal.raw_value[:200],
                topics=proposal.topics,
                target_recipients=target_recipients,
            )
            self._items[memory_id] = item
            self._proposal_to_memory[proposal.proposal_id] = memory_id
            if update_target is not None:
                old_item = self._items[update_target]
                self._items[update_target] = MemoryItem(
                    memory_id=old_item.memory_id,
                    proposal_id=old_item.proposal_id,
                    task_id=old_item.task_id,
                    title=old_item.title,
                    raw_value=old_item.raw_value,
                    visibility=old_item.visibility,
                    source_agent=old_item.source_agent,
                    source=old_item.source,
                    step_index=old_item.step_index,
                    active=False,
                    supersedes=old_item.supersedes,
                    update=old_item.update,
                    created_at=old_item.created_at,
                    summary=old_item.summary,
                    topics=old_item.topics,
                    target_recipients=getattr(old_item, "target_recipients", ()),
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
                    or reader_id in getattr(item, "target_recipients", ())
                )
            ]

    def read(self, memory_id: str, reader_id: str, task_id: str) -> MemoryItem:
        with self._lock:
            item = self._items[memory_id]
            if not item.active or item.task_id != task_id:
                raise KeyError(memory_id)
            is_allowed = (
                item.visibility == "global"
                or reader_id in getattr(item, "target_recipients", ())
            )
            if not is_allowed:
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

    _validate_update = _validate_supersession
