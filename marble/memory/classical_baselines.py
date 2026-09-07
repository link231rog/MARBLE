"""Adapters for classical memory baselines: Mem0, A-Mem, and MemoryOS.

These adapters implement the uniform GovernedMemory/MemoryStep lifecycle interface:
- submit(proposal, **metadata)
- visible_keys(reader_id, task_id, query, top_k)
- read(memory_id, reader_id, task_id)
- bank.all_items()
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from .schema import MemoryCard, MemoryProposal
from .trace import TraceLogger


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


# ==============================================================================
# 1. Mem0 (Chhikara et al., 2025): In-Context Fact Extraction & Global Vector CRUD
# ==============================================================================

@dataclass
class Mem0Item:
    memory_id: str
    task_id: str
    title: str
    raw_value: str
    source_agent: str
    step_index: int
    active: bool = True
    visibility: str = "global"
    created_at: int = 0
    updated_at: int = 0


class _Mem0BankView:
    def __init__(self, items_dict: Dict[str, Mem0Item]) -> None:
        self._items = items_dict

    def all_items(self) -> List[Mem0Item]:
        return [item for item in self._items.values() if item.active]


class Mem0Adapter:
    """Mem0-style memory: Task-scoped global memory with in-context CRUD operations.

    Mem0 uses in-context LLM instructions to choose {ADD, UPDATE, DELETE, NOOP}
    against existing memories, without RL tuning or answer-side distillation.
    """

    def __init__(
        self,
        trace: Optional[TraceLogger] = None,
        manager_fn: Optional[Callable[[MemoryProposal, Sequence[Mem0Item]], Dict[str, Any]]] = None,
    ) -> None:
        self._items: Dict[str, Mem0Item] = {}
        self._clock = 0
        self._lock = RLock()
        self.trace = trace
        self.controller = None
        self.bank = _Mem0BankView(self._items)
        self.manager_fn = manager_fn or self._default_manager

    def _default_manager(
        self, proposal: MemoryProposal, active: Sequence[Mem0Item]
    ) -> Dict[str, Any]:
        if not proposal.raw_value.strip():
            return {"operation": "NOOP", "memory_id": None}
        prop_tokens = _tokens(proposal.title + " " + proposal.raw_value)
        # Search for strong lexical match for UPDATE
        for item in active:
            item_tokens = _tokens(item.title + " " + item.raw_value)
            if item.title.strip().casefold() == proposal.title.strip().casefold():
                return {"operation": "UPDATE", "memory_id": item.memory_id}
            if prop_tokens and len(prop_tokens & item_tokens) / max(len(prop_tokens), 1) > 0.8:
                return {"operation": "UPDATE", "memory_id": item.memory_id}
        return {"operation": "ADD", "memory_id": None}

    def submit(self, proposal: MemoryProposal, **metadata) -> Optional[Mem0Item]:
        with self._lock:
            active = [item for item in self._items.values() if item.task_id == proposal.task_id and item.active]
            decision = self.manager_fn(proposal, active)
            op = str(decision.get("operation", "NOOP")).upper()
            target_id = decision.get("memory_id")
            self._clock += 1

            res_item: Optional[Mem0Item] = None
            if op == "ADD":
                item_id = proposal.proposal_id or f"mem0-{self._clock}"
                res_item = Mem0Item(
                    memory_id=item_id,
                    task_id=proposal.task_id,
                    title=proposal.title,
                    raw_value=proposal.raw_value,
                    source_agent=proposal.agent_id,
                    step_index=proposal.step_index,
                    active=True,
                    created_at=self._clock,
                    updated_at=self._clock,
                )
                self._items[item_id] = res_item
            elif op == "UPDATE" and target_id and target_id in self._items:
                old = self._items[target_id]
                res_item = Mem0Item(
                    memory_id=old.memory_id,
                    task_id=old.task_id,
                    title=proposal.title,
                    raw_value=proposal.raw_value,
                    source_agent=proposal.agent_id,
                    step_index=proposal.step_index,
                    active=True,
                    created_at=old.created_at,
                    updated_at=self._clock,
                )
                self._items[target_id] = res_item
            elif op == "DELETE" and target_id and target_id in self._items:
                self._items[target_id].active = False
                res_item = self._items[target_id]

            if self.trace is not None:
                self.trace.log(
                    "classical_memory_operation",
                    operation=op,
                    memory_id=target_id or (res_item.memory_id if res_item else None),
                    task_id=proposal.task_id,
                    agent_id=proposal.agent_id,
                    proposal=proposal.__dict__,
                )
            return res_item if res_item and res_item.active else None

    def visible_keys(
        self,
        reader_id: str,
        task_id: str,
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryCard]:
        del reader_id
        with self._lock:
            active = [item for item in self._items.values() if item.task_id == task_id and item.active]
            if not query:
                ranked = sorted(active, key=lambda x: -x.updated_at)[:top_k]
            else:
                q_tok = _tokens(query)
                scored = [
                    (len(q_tok & _tokens(item.title + " " + item.raw_value)), item.updated_at, item)
                    for item in active
                ]
                scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
                ranked = [item for _, _, item in scored[:top_k]]

            return [
                MemoryCard(
                    memory_id=item.memory_id,
                    task_id=item.task_id,
                    title=item.title,
                    visibility="global",
                )
                for item in ranked
            ]

    def read(self, memory_id: str, reader_id: str, task_id: str) -> Mem0Item:
        with self._lock:
            item = self._items[memory_id]
            if self.trace is not None:
                self.trace.log_read(memory_id, reader_id=reader_id, task_id=task_id)
            return item


# ==============================================================================
# 2. A-Mem (Xu et al., NeurIPS 2025): Zettelkasten Dynamic Note Linking & Evolution
# ==============================================================================

@dataclass
class AMemNote:
    memory_id: str
    task_id: str
    title: str
    raw_value: str
    source_agent: str
    step_index: int
    tags: Set[str] = field(default_factory=set)
    links: Set[str] = field(default_factory=set)
    active: bool = True
    visibility: str = "global"
    created_at: int = 0
    updated_at: int = 0


class _AMemBankView:
    def __init__(self, notes_dict: Dict[str, AMemNote]) -> None:
        self._notes = notes_dict

    def all_items(self) -> List[AMemNote]:
        return [note for note in self._notes.values() if note.active]


class AMemAdapter:
    """A-Mem style memory: Zettelkasten-inspired note linking and recursive evolution.

    When new notes arrive, links are established to existing notes with high semantic
    overlap. When retrieving notes, it expands to 1-hop linked neighbors ("boxes").
    """

    def __init__(
        self,
        trace: Optional[TraceLogger] = None,
        max_links_per_note: int = 3,
    ) -> None:
        self._notes: Dict[str, AMemNote] = {}
        self._clock = 0
        self._lock = RLock()
        self.trace = trace
        self.controller = None
        self.bank = _AMemBankView(self._notes)
        self.max_links_per_note = max_links_per_note

    def submit(self, proposal: MemoryProposal, **metadata) -> Optional[AMemNote]:
        if not proposal.raw_value.strip():
            return None
        with self._lock:
            self._clock += 1
            note_id = proposal.proposal_id or f"amem-{self._clock}"
            prop_tok = _tokens(proposal.title + " " + proposal.raw_value)
            tags = {tok for tok in prop_tok if len(tok) > 4}

            # 1. Note creation
            note = AMemNote(
                memory_id=note_id,
                task_id=proposal.task_id,
                title=proposal.title,
                raw_value=proposal.raw_value,
                source_agent=proposal.agent_id,
                step_index=proposal.step_index,
                tags=tags,
                active=True,
                created_at=self._clock,
                updated_at=self._clock,
            )

            # 2. Dynamic Link Generation: find top relevant existing notes
            active_notes = [n for n in self._notes.values() if n.task_id == proposal.task_id and n.active]
            scored_links = []
            for other in active_notes:
                other_tok = _tokens(other.title + " " + other.raw_value)
                overlap = len(prop_tok & other_tok)
                if overlap > 0:
                    scored_links.append((overlap, other))
            scored_links.sort(key=lambda x: x[0], reverse=True)

            # Link top neighbors
            for _, neighbor in scored_links[: self.max_links_per_note]:
                note.links.add(neighbor.memory_id)
                neighbor.links.add(note_id)
                # 3. Evolution: evolve neighbor tags
                neighbor.tags.update(list(tags)[:2])
                neighbor.updated_at = self._clock

            self._notes[note_id] = note
            if self.trace is not None:
                self.trace.log(
                    "classical_memory_operation",
                    operation="ADD",
                    memory_id=note_id,
                    task_id=proposal.task_id,
                    agent_id=proposal.agent_id,
                    proposal=proposal.__dict__,
                    links=list(note.links),
                )
            return note

    def visible_keys(
        self,
        reader_id: str,
        task_id: str,
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryCard]:
        del reader_id
        with self._lock:
            active = [n for n in self._notes.values() if n.task_id == task_id and n.active]
            if not active:
                return []

            # Primary retrieval
            if not query:
                primary = sorted(active, key=lambda x: -x.updated_at)[:top_k]
            else:
                q_tok = _tokens(query)
                scored = [
                    (len(q_tok & _tokens(n.title + " " + n.raw_value)), n.updated_at, n)
                    for n in active
                ]
                scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
                primary = [n for _, _, n in scored[: max(1, top_k // 2)]]

            # Box expansion: include 1-hop linked neighbors (A-Mem feature)
            selected_ids = {n.memory_id for n in primary}
            result_notes = list(primary)
            for n in primary:
                for link_id in n.links:
                    if link_id in self._notes and link_id not in selected_ids:
                        selected_ids.add(link_id)
                        result_notes.append(self._notes[link_id])
                        if len(result_notes) >= top_k:
                            break
                if len(result_notes) >= top_k:
                    break

            return [
                MemoryCard(
                    memory_id=n.memory_id,
                    task_id=n.task_id,
                    title=n.title,
                    visibility="global",
                )
                for n in result_notes[:top_k]
            ]

    def read(self, memory_id: str, reader_id: str, task_id: str) -> AMemNote:
        with self._lock:
            note = self._notes[memory_id]
            if self.trace is not None:
                self.trace.log_read(memory_id, reader_id=reader_id, task_id=task_id)
            return note


# ==============================================================================
# 3. MemoryOS (Kang et al., 2025): OS Hierarchical Paging (STM, MTM, LTM) & Heat Eviction
# ==============================================================================

@dataclass
class MemoryOSItem:
    memory_id: str
    task_id: str
    title: str
    raw_value: str
    source_agent: str
    step_index: int
    tier: str  # "STM", "MTM", "LTM"
    heat: float = 1.0
    active: bool = True
    visibility: str = "global"
    created_at: int = 0
    updated_at: int = 0


class _MemoryOSBankView:
    def __init__(self, items_dict: Dict[str, MemoryOSItem]) -> None:
        self._items = items_dict

    def all_items(self) -> List[MemoryOSItem]:
        return [item for item in self._items.values() if item.active]


class MemoryOSAdapter:
    """MemoryOS-style memory: 3-tier hierarchy (STM, MTM, LTM) with OS heat-based eviction.

    - STM: FIFO short-term buffer (fixed capacity).
    - MTM: Mid-term working pages (capacity M), evicted by lowest heat (access count + recency).
    - LTM: Long-term archival store for demoted pages.
    """

    def __init__(
        self,
        trace: Optional[TraceLogger] = None,
        stm_capacity: int = 4,
        mtm_capacity: int = 10,
    ) -> None:
        self._items: Dict[str, MemoryOSItem] = {}
        self._stm_queue: collections.deque[str] = collections.deque()
        self._clock = 0
        self._lock = RLock()
        self.trace = trace
        self.controller = None
        self.bank = _MemoryOSBankView(self._items)
        self.stm_capacity = stm_capacity
        self.mtm_capacity = mtm_capacity

    def submit(self, proposal: MemoryProposal, **metadata) -> Optional[MemoryOSItem]:
        if not proposal.raw_value.strip():
            return None
        with self._lock:
            self._clock += 1
            item_id = proposal.proposal_id or f"memos-{self._clock}"

            item = MemoryOSItem(
                memory_id=item_id,
                task_id=proposal.task_id,
                title=proposal.title,
                raw_value=proposal.raw_value,
                source_agent=proposal.agent_id,
                step_index=proposal.step_index,
                tier="STM",
                heat=1.0,
                active=True,
                created_at=self._clock,
                updated_at=self._clock,
            )
            self._items[item_id] = item

            # FIFO into STM
            self._stm_queue.append(item_id)
            if len(self._stm_queue) > self.stm_capacity:
                demoted_id = self._stm_queue.popleft()
                if demoted_id in self._items:
                    self._items[demoted_id].tier = "MTM"
                    self._check_mtm_capacity(proposal.task_id)

            if self.trace is not None:
                self.trace.log(
                    "classical_memory_operation",
                    operation="ADD",
                    memory_id=item_id,
                    task_id=proposal.task_id,
                    agent_id=proposal.agent_id,
                    proposal=proposal.__dict__,
                    tier="STM",
                )
            return item

    def _check_mtm_capacity(self, task_id: str) -> None:
        """Evict lowest-heat MTM items to LTM if MTM exceeds capacity."""
        mtm_items = [
            item for item in self._items.values()
            if item.task_id == task_id and item.active and item.tier == "MTM"
        ]
        if len(mtm_items) > self.mtm_capacity:
            mtm_items.sort(key=lambda x: x.heat)
            to_evict = mtm_items[: len(mtm_items) - self.mtm_capacity]
            for it in to_evict:
                it.tier = "LTM"

    def visible_keys(
        self,
        reader_id: str,
        task_id: str,
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryCard]:
        del reader_id
        with self._lock:
            active = [item for item in self._items.values() if item.task_id == task_id and item.active]
            if not active:
                return []

            # MemoryOS scores: relevance + heat * 0.5 + tier_bonus
            tier_weights = {"STM": 1.5, "MTM": 1.2, "LTM": 0.5}
            q_tok = _tokens(query) if query else set()

            scored = []
            for item in active:
                overlap = len(q_tok & _tokens(item.title + " " + item.raw_value)) if q_tok else 1
                score = overlap + 0.3 * item.heat + tier_weights.get(item.tier, 1.0)
                scored.append((score, item))

            scored.sort(key=lambda s: s[0], reverse=True)
            ranked = [item for _, item in scored[:top_k]]

            # Hit increases heat
            for item in ranked:
                item.heat += 0.5

            return [
                MemoryCard(
                    memory_id=item.memory_id,
                    task_id=item.task_id,
                    title=item.title,
                    visibility="global",
                )
                for item in ranked
            ]

    def read(self, memory_id: str, reader_id: str, task_id: str) -> MemoryOSItem:
        with self._lock:
            item = self._items[memory_id]
            item.heat += 1.0  # Access heat boost
            if self.trace is not None:
                self.trace.log_read(memory_id, reader_id=reader_id, task_id=task_id)
            return item
