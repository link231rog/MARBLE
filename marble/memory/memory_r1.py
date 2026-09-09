from __future__ import annotations

import re
from dataclasses import dataclass
from threading import RLock
from typing import Dict, List, Literal, Optional

from .schema import MemoryProposal

MemoryR1Operation = Literal["ADD", "UPDATE", "DELETE", "NOOP"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


@dataclass(frozen=True)
class MemoryR1Card:
    """A compact key card for deterministic global-memory retrieval."""

    memory_id: str
    task_id: str
    title: str


@dataclass(frozen=True)
class MemoryR1Item:
    """A task-scoped item in the Memory-R1-style global store."""

    memory_id: str
    task_id: str
    title: str
    raw_value: str
    source_agent: str
    source: str
    step_index: int
    active: bool
    created_at: int
    updated_at: int
    visibility: str = "global"

    def card(self) -> MemoryR1Card:
        return MemoryR1Card(
            memory_id=self.memory_id,
            task_id=self.task_id,
            title=self.title,
        )


@dataclass(frozen=True)
class MemoryR1TraceRecord:
    """Structured local trace emitted by every memory operation."""

    operation: MemoryR1Operation
    memory_id: Optional[str]
    task_id: str
    title: Optional[str] = None
    raw_value: Optional[str] = None


@dataclass(frozen=True)
class MemoryR1OperationResult:
    item: Optional[MemoryR1Item]
    trace: MemoryR1TraceRecord


class MemoryR1Memory:
    """Pure local global CRUD memory for the adapted Memory-R1 baseline.

    The store intentionally has no private/global visibility schema. Every
    active item is globally retrievable inside its task scope.
    """

    def __init__(self) -> None:
        self._items: Dict[str, MemoryR1Item] = {}
        self._clock = 0
        self._lock = RLock()

    def add(
        self,
        proposal: MemoryProposal,
        memory_id: Optional[str] = None,
    ) -> MemoryR1OperationResult:
        self._validate_task_id(proposal.task_id)
        item_id = memory_id or proposal.proposal_id

        with self._lock:
            if item_id in self._items:
                raise ValueError("memory_id has already been used")
            timestamp = self._tick()
            item = MemoryR1Item(
                memory_id=item_id,
                task_id=proposal.task_id,
                title=proposal.title,
                raw_value=proposal.raw_value,
                source_agent=proposal.agent_id,
                source=proposal.source,
                step_index=proposal.step_index,
                active=True,
                created_at=timestamp,
                updated_at=timestamp,
            )
            self._items[item_id] = item
            return self._result("ADD", item)

    def update(
        self,
        memory_id: str,
        proposal: MemoryProposal,
    ) -> MemoryR1OperationResult:
        self._validate_task_id(proposal.task_id)

        with self._lock:
            old_item = self._active_item(memory_id, proposal.task_id)
            timestamp = self._tick()
            item = MemoryR1Item(
                memory_id=old_item.memory_id,
                task_id=old_item.task_id,
                title=proposal.title,
                raw_value=proposal.raw_value,
                source_agent=proposal.agent_id,
                source=proposal.source,
                step_index=proposal.step_index,
                active=True,
                created_at=old_item.created_at,
                updated_at=timestamp,
            )
            self._items[memory_id] = item
            return self._result("UPDATE", item)

    def delete(self, memory_id: str, task_id: str) -> MemoryR1OperationResult:
        self._validate_task_id(task_id)

        with self._lock:
            old_item = self._active_item(memory_id, task_id)
            timestamp = self._tick()
            item = MemoryR1Item(
                memory_id=old_item.memory_id,
                task_id=old_item.task_id,
                title=old_item.title,
                raw_value=old_item.raw_value,
                source_agent=old_item.source_agent,
                source=old_item.source,
                step_index=old_item.step_index,
                active=False,
                created_at=old_item.created_at,
                updated_at=timestamp,
            )
            self._items[memory_id] = item
            return self._result("DELETE", item)

    def noop(
        self,
        task_id: str,
        memory_id: Optional[str] = None,
    ) -> MemoryR1OperationResult:
        self._validate_task_id(task_id)

        with self._lock:
            if memory_id is not None:
                self._item_in_task(memory_id, task_id)
            trace = MemoryR1TraceRecord(
                operation="NOOP",
                memory_id=memory_id,
                task_id=task_id,
            )
            return MemoryR1OperationResult(item=None, trace=trace)

    def active_items(self, task_id: str) -> List[MemoryR1Item]:
        self._validate_task_id(task_id)
        with self._lock:
            return [
                item
                for item in self._items.values()
                if item.task_id == task_id and item.active
            ]

    def retrieve(
        self,
        task_id: str,
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryR1Card]:
        if top_k <= 0:
            return []
        cards = [item.card() for item in self.active_items(task_id)]
        if not query:
            return cards[:top_k]

        query_tokens = _tokens(query)
        if not query_tokens:
            return cards[:top_k]
        scored = [
            (-len(query_tokens & _tokens(card.title)), index, card)
            for index, card in enumerate(cards)
        ]
        scored.sort(key=lambda entry: (entry[0], entry[1]))
        return [card for _, _, card in scored[:top_k]]

    def read(self, memory_id: str, task_id: str) -> MemoryR1Item:
        self._validate_task_id(task_id)
        with self._lock:
            return self._active_item(memory_id, task_id)

    def get(self, memory_id: str) -> Optional[MemoryR1Item]:
        with self._lock:
            return self._items.get(memory_id)

    def _result(
        self,
        operation: MemoryR1Operation,
        item: MemoryR1Item,
    ) -> MemoryR1OperationResult:
        return MemoryR1OperationResult(
            item=item,
            trace=MemoryR1TraceRecord(
                operation=operation,
                memory_id=item.memory_id,
                task_id=item.task_id,
                title=item.title,
                raw_value=item.raw_value,
            ),
        )

    def _active_item(self, memory_id: str, task_id: str) -> MemoryR1Item:
        item = self._item_in_task(memory_id, task_id)
        if not item.active:
            raise KeyError(memory_id)
        return item

    def _item_in_task(self, memory_id: str, task_id: str) -> MemoryR1Item:
        item = self._items.get(memory_id)
        if item is None or item.task_id != task_id:
            raise KeyError(memory_id)
        return item

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        if not task_id:
            raise ValueError("task_id must not be empty")
