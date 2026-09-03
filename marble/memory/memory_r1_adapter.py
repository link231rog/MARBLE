"""Adapter for running the adapted Memory-R1 baseline in the MARBLE engine."""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Sequence

from .memory_r1 import MemoryR1Item, MemoryR1Memory
from .rewards import token_count
from .schema import MemoryCard, MemoryProposal
from .trace import TraceLogger

CrudManager = Callable[[MemoryProposal, Sequence[MemoryR1Item]], Dict[str, object]]
DistillFn = Callable[[str, List[str]], str]


class _R1BankView:
    """Small bank-shaped view consumed by the shared MemoryStep harness."""

    def __init__(self, memory: MemoryR1Memory) -> None:
        self._memory = memory

    def all_items(self) -> List[MemoryR1Item]:
        task_ids = sorted({item.task_id for item in self._memory._items.values()})
        return [
            item
            for task_id in task_ids
            for item in self._memory.active_items(task_id)
        ]


class MemoryR1Adapter:
    """Expose Memory-R1 CRUD through the governed-memory lifecycle.

    New worker outputs are ADDed; a repeated title becomes UPDATE; empty output
    is NOOP. DELETE remains available on the manager for the CRUD baseline.
    """

    def __init__(
        self,
        memory: Optional[MemoryR1Memory] = None,
        trace: Optional[TraceLogger] = None,
        manager: Optional[CrudManager] = None,
        distill_fn: Optional[DistillFn] = None,
    ) -> None:
        self.memory = memory or MemoryR1Memory()
        self.bank = _R1BankView(self.memory)
        self.trace = trace
        self.controller = None
        self.manager = manager or _default_manager
        self.distill_fn = distill_fn

    def submit(self, proposal: MemoryProposal, **metadata):
        active = self.memory.active_items(proposal.task_id)
        decision = self.manager(proposal, active)
        operation = str(decision.get("operation", "NOOP")).upper()
        memory_id = decision.get("memory_id")
        if operation == "ADD":
            result = self.memory.add(proposal)
        elif operation == "UPDATE" and isinstance(memory_id, str):
            result = self.memory.update(memory_id, proposal)
        elif operation == "DELETE" and isinstance(memory_id, str):
            result = self.memory.delete(memory_id, proposal.task_id)
        else:
            result = self.memory.noop(proposal.task_id)
        self._log(result.trace, proposal, metadata, decision)
        return result.item if result.item is not None and result.item.active else None

    def visible_keys(
        self,
        reader_id: str,
        task_id: str,
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryCard]:
        del reader_id
        return [
            MemoryCard(
                memory_id=card.memory_id,
                task_id=card.task_id,
                title=card.title,
                visibility="global",
                owner_id=None,
            )
            for card in self.memory.retrieve(task_id, query=query, top_k=top_k)
        ]

    def read(self, memory_id: str, reader_id: str, task_id: str) -> MemoryR1Item:
        item = self.memory.read(memory_id, task_id)
        if self.trace is not None:
            self.trace.log_read(memory_id, reader_id=reader_id, task_id=task_id)
        return item

    def distill(self, task_text: str, notes: List[str]) -> List[str]:
        """Memory-R1-only answer-side distillation of already read evidence."""
        if self.distill_fn is None or not notes:
            return notes
        output = self.distill_fn(task_text, notes).strip()
        if self.trace is not None:
            self.trace.log(
                "memory_r1_distillation",
                input_tokens=token_count("\n".join(notes)),
                output_tokens=token_count(output),
                api_calls=1,
            )
        return [output] if output else []

    def _log(self, operation, proposal: MemoryProposal, metadata, decision) -> None:
        if self.trace is None:
            return
        self.trace.log(
            "memory_r1_operation",
            operation=operation.operation,
            memory_id=operation.memory_id,
            task_id=operation.task_id,
            proposal=proposal.__dict__,
            **{
                key: value
                for key, value in metadata.items()
                if key in {"memory_card_tokens", "injected_memory_tokens"}
            },
            **{
                key: value
                for key, value in decision.items()
                if key.startswith("manager_")
            },
        )


def _default_manager(
    proposal: MemoryProposal,
    active: Sequence[MemoryR1Item],
) -> Dict[str, object]:
    """Deterministic fallback for offline tests and no-manager configurations."""
    if not proposal.raw_value.strip():
        return {"operation": "NOOP", "memory_id": None}
    title = proposal.title.strip().casefold()
    for item in active:
        if item.title.strip().casefold() == title:
            return {"operation": "UPDATE", "memory_id": item.memory_id}
    return {"operation": "ADD", "memory_id": None}


def parse_crud_decision(
    raw: str,
    active: Sequence[MemoryR1Item],
) -> Dict[str, object]:
    """Robustly validate an LLM CRUD decision; tolerant to markdown fences and extra fields."""
    text = (raw or "").strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        match_brace = re.search(r"\{[^{}]*\"operation\"[^{}]*\}", text, re.DOTALL)
        if match_brace:
            text = match_brace.group(0)

    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return {"operation": "NOOP", "memory_id": None}

    if not isinstance(parsed, dict):
        return {"operation": "NOOP", "memory_id": None}

    operation = str(parsed.get("operation", "NOOP")).strip().upper()
    if operation not in {"ADD", "UPDATE", "DELETE", "NOOP"}:
        return {"operation": "NOOP", "memory_id": None}

    memory_id = parsed.get("memory_id")
    if memory_id is not None:
        memory_id = str(memory_id).strip()
        if memory_id.lower() in {"null", "none", ""}:
            memory_id = None

    active_ids = {item.memory_id for item in active}
    if operation in {"UPDATE", "DELETE"}:
        if not memory_id or memory_id not in active_ids:
            return {"operation": "NOOP", "memory_id": None}
    elif operation in {"ADD", "NOOP"}:
        memory_id = None

    return {"operation": operation, "memory_id": memory_id}
