"""Dedicated Multi-Agent Memory (MAM) Baselines (2025-2026):
1. G-Memory (NeurIPS 2025): Three-tier hierarchical graph memory (Insight, Query, Interaction).
2. CollabMem (ICML 2025/2026): Bipartite access graph with asymmetric tiers (Private, Group, Public).
3. COPPER (NeurIPS 2024/2025): Reflective multi-agent collaboration via counterfactual memory.

All adapters implement the uniform GovernedMemory lifecycle interface:
- submit(proposal, **metadata)
- visible_keys(reader_id, task_id, query, top_k)
- read(memory_id, reader_id, task_id)
- bank.all_items()
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, List, Optional, Sequence, Set

from .rewards import token_count
from .schema import MemoryCard, MemoryProposal
from .trace import TraceLogger

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> Set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


# ==============================================================================
# 1. G-Memory (NeurIPS 2025): Hierarchical Graph Memory (Insight / Query / Interaction)
# ==============================================================================

@dataclass
class GMemoryItem:
    memory_id: str
    task_id: str
    title: str
    raw_value: str
    source_agent: str
    step_index: int
    tier: str  # "insight", "query", "interaction"
    active: bool = True
    visibility: str = "global"
    created_at: int = 0
    updated_at: int = 0
    linked_ids: List[str] = field(default_factory=list)


class _GMemoryBankView:
    def __init__(self, items_dict: Dict[str, GMemoryItem]) -> None:
        self._items = items_dict

    def all_items(self) -> List[GMemoryItem]:
        return [item for item in self._items.values() if item.active]


class GMemoryAdapter:
    """G-Memory (NeurIPS 2025) baseline adapter:
    Hierarchical 3-tier memory graph linking high-level Insights, Query subgoals,
    and granular Interaction trajectories.
    """

    def __init__(self, trace: Optional[TraceLogger] = None) -> None:
        self._items: Dict[str, GMemoryItem] = {}
        self._clock = 0
        self._lock = RLock()
        self.trace = trace
        self.controller = None
        self.bank = _GMemoryBankView(self._items)

    def _classify_tier(self, proposal: MemoryProposal) -> str:
        text = (proposal.title + " " + proposal.raw_value).lower()
        if any(w in text for w in ("root cause is", "decision", "synthesis", "confirmed", "conclusion", "framework", "final")):
            return "insight"
        if any(w in text for w in ("investigate", "explore", "task", "plan", "assign", "check", "hypothesis")):
            return "query"
        return "interaction"

    def submit(self, proposal: MemoryProposal, **metadata) -> Optional[GMemoryItem]:
        with self._lock:
            if not proposal.raw_value.strip():
                return None
            self._clock += 1
            tier = self._classify_tier(proposal)
            item_id = proposal.proposal_id or f"gmem-{self._clock}"

            # Link to existing items of adjacent tiers (bi-directional graph edges)
            linked: List[str] = []
            for existing_id, item in self._items.items():
                if item.task_id == proposal.task_id and item.active:
                    if tier == "insight" and item.tier == "query":
                        linked.append(existing_id)
                    elif tier == "query" and item.tier == "interaction":
                        linked.append(existing_id)

            item = GMemoryItem(
                memory_id=item_id,
                task_id=proposal.task_id,
                title=proposal.title,
                raw_value=proposal.raw_value,
                source_agent=proposal.agent_id,
                step_index=proposal.step_index,
                tier=tier,
                active=True,
                created_at=self._clock,
                updated_at=self._clock,
                linked_ids=linked[:5],
            )
            self._items[item_id] = item

            if self.trace is not None:
                self.trace.log(
                    "g_memory_operation",
                    tier=tier,
                    memory_id=item_id,
                    task_id=proposal.task_id,
                    agent_id=proposal.agent_id,
                    proposal=proposal.__dict__,
                    linked_count=len(linked),
                )
            return item

    def visible_keys(
        self,
        reader_id: str,
        task_id: str,
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryCard]:
        del reader_id
        with self._lock:
            active = [it for it in self._items.values() if it.task_id == task_id and it.active]
            if not active:
                return []

            q_tok = _tokens(query) if query else set()

            # Hierarchical traversal scoring: Insights get +2.0 bonus, Query gets +1.0 bonus
            scored = []
            for it in active:
                overlap = len(q_tok & _tokens(it.title + " " + it.raw_value)) if q_tok else 1
                tier_bonus = 2.0 if it.tier == "insight" else (1.0 if it.tier == "query" else 0.2)
                score = overlap * 1.5 + tier_bonus + (it.updated_at / (self._clock + 1)) * 0.5
                scored.append((score, it))

            scored.sort(key=lambda s: s[0], reverse=True)
            ranked = [it for _, it in scored[:top_k]]

            return [
                MemoryCard(
                    memory_id=it.memory_id,
                    task_id=it.task_id,
                    title=f"[{it.tier.upper()}] {it.title}",
                    visibility="global",
                )
                for it in ranked
            ]

    def read(self, memory_id: str, reader_id: str, task_id: str) -> GMemoryItem:
        with self._lock:
            item = self._items[memory_id]
            if self.trace is not None:
                self.trace.log_read(memory_id, reader_id=reader_id, task_id=task_id)
            return item


# ==============================================================================
# 2. CollabMem (ICML 2025/2026): Bipartite Access Graph & Tiered Isolation
# ==============================================================================

@dataclass
class CollabMemItem:
    memory_id: str
    task_id: str
    title: str
    raw_value: str
    source_agent: str
    step_index: int
    access_tier: str  # "private", "group", "public"
    allowed_agents: Set[str]
    active: bool = True
    visibility: str = "global"
    created_at: int = 0
    updated_at: int = 0


class _CollabMemBankView:
    def __init__(self, items_dict: Dict[str, CollabMemItem]) -> None:
        self._items = items_dict

    def all_items(self) -> List[CollabMemItem]:
        return [item for item in self._items.values() if item.active]


class CollabMemAdapter:
    """CollabMem (ICML 2025/2026) baseline adapter:
    Bipartite access control enforcing Private, Group, and Public access scopes
    to resolve the multi-agent privacy-collaboration dilemma.
    """

    def __init__(self, trace: Optional[TraceLogger] = None) -> None:
        self._items: Dict[str, CollabMemItem] = {}
        self._clock = 0
        self._lock = RLock()
        self.trace = trace
        self.controller = None
        self.bank = _CollabMemBankView(self._items)

    def _determine_scope(self, proposal: MemoryProposal) -> tuple[str, Set[str]]:
        text = (proposal.title + " " + proposal.raw_value).lower()
        # Public scope: verified consensus or core team findings
        if any(w in text for w in ("decision", "root cause", "summary", "confirmed", "consensus", "final answer")):
            return "public", set()

        # Group scope: cross-agent coordination mentioning other agents or tasks
        mentioned = set(re.findall(r"agent\d+", text))
        mentioned.add(proposal.agent_id)
        if len(mentioned) > 1 or any(w in text for w in ("collaborate", "hand off", "assign", "team")):
            return "group", mentioned

        # Private scope: intermediate local exploration
        return "private", {proposal.agent_id}

    def submit(self, proposal: MemoryProposal, **metadata) -> Optional[CollabMemItem]:
        with self._lock:
            if not proposal.raw_value.strip():
                return None
            self._clock += 1
            scope, allowed = self._determine_scope(proposal)
            item_id = proposal.proposal_id or f"collab-{self._clock}"

            item = CollabMemItem(
                memory_id=item_id,
                task_id=proposal.task_id,
                title=proposal.title,
                raw_value=proposal.raw_value,
                source_agent=proposal.agent_id,
                step_index=proposal.step_index,
                access_tier=scope,
                allowed_agents=allowed,
                active=True,
                created_at=self._clock,
                updated_at=self._clock,
            )
            self._items[item_id] = item

            if self.trace is not None:
                self.trace.log(
                    "collabmem_operation",
                    access_tier=scope,
                    allowed_agents=list(allowed),
                    memory_id=item_id,
                    task_id=proposal.task_id,
                    agent_id=proposal.agent_id,
                    proposal=proposal.__dict__,
                )
            return item

    def visible_keys(
        self,
        reader_id: str,
        task_id: str,
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryCard]:
        with self._lock:
            # Bipartite Access Graph check:
            # Allowed if public, or reader is in allowed_agents
            authorized = [
                it for it in self._items.values()
                if it.task_id == task_id and it.active and (
                    it.access_tier == "public" or reader_id in it.allowed_agents
                )
            ]
            if not authorized:
                return []

            q_tok = _tokens(query) if query else set()
            scored = []
            for it in authorized:
                overlap = len(q_tok & _tokens(it.title + " " + it.raw_value)) if q_tok else 1
                scope_weight = 1.5 if it.access_tier == "public" else (1.2 if it.access_tier == "group" else 1.0)
                score = overlap * scope_weight + (it.updated_at / (self._clock + 1)) * 0.3
                scored.append((score, it))

            scored.sort(key=lambda s: s[0], reverse=True)
            ranked = [it for _, it in scored[:top_k]]

            return [
                MemoryCard(
                    memory_id=it.memory_id,
                    task_id=it.task_id,
                    title=f"[{it.access_tier.upper()}] {it.title}",
                    visibility="global" if it.access_tier == "public" else "targeted",
                    target_recipients=() if it.access_tier == "public" else tuple(sorted(it.allowed_agents)),
                )
                for it in ranked
            ]

    def read(self, memory_id: str, reader_id: str, task_id: str) -> CollabMemItem:
        with self._lock:
            item = self._items[memory_id]
            if self.trace is not None:
                self.trace.log_read(memory_id, reader_id=reader_id, task_id=task_id)
            return item


# ==============================================================================
# 3. COPPER (NeurIPS 2024/2025): Counterfactual Reflective Shared Memory
# ==============================================================================

@dataclass
class COPPERItem:
    memory_id: str
    task_id: str
    title: str
    raw_value: str
    source_agent: str
    step_index: int
    is_reflection: bool = False
    active: bool = True
    visibility: str = "global"
    created_at: int = 0
    updated_at: int = 0


class _COPPERBankView:
    def __init__(self, items_dict: Dict[str, COPPERItem]) -> None:
        self._items = items_dict

    def all_items(self) -> List[COPPERItem]:
        return [item for item in self._items.values() if item.active]


class COPPERAdapter:
    """COPPER (NeurIPS 2024/2025) baseline adapter:
    Augments shared factual memory with Counterfactual Reflection entries
    derived from execution feedback to guide multi-agent course-correction.
    """

    def __init__(self, trace: Optional[TraceLogger] = None) -> None:
        self._items: Dict[str, COPPERItem] = {}
        self._clock = 0
        self._lock = RLock()
        self.trace = trace
        self.controller = None
        self.bank = _COPPERBankView(self._items)

    def submit(self, proposal: MemoryProposal, **metadata) -> Optional[COPPERItem]:
        with self._lock:
            if not proposal.raw_value.strip():
                return None
            self._clock += 1
            item_id = proposal.proposal_id or f"copper-{self._clock}"

            base_item = COPPERItem(
                memory_id=item_id,
                task_id=proposal.task_id,
                title=proposal.title,
                raw_value=proposal.raw_value,
                source_agent=proposal.agent_id,
                step_index=proposal.step_index,
                is_reflection=False,
                active=True,
                created_at=self._clock,
                updated_at=self._clock,
            )
            self._items[item_id] = base_item

            # Causal counterfactual reflection generation (COPPER mechanism)
            text = (proposal.title + " " + proposal.raw_value).lower()
            if any(w in text for w in ("error", "does not exist", "zero", "not present", "none", "failed")):
                self._clock += 1
                ref_id = f"copper-refl-{self._clock}"
                ref_item = COPPERItem(
                    memory_id=ref_id,
                    task_id=proposal.task_id,
                    title=f"[Reflect: Avoid] {proposal.title[:30]}",
                    raw_value=f"Counterfactual Lesson from {proposal.agent_id}: Path yielded null/negative results. Pivot to alternate root causes.",
                    source_agent=proposal.agent_id,
                    step_index=proposal.step_index,
                    is_reflection=True,
                    active=True,
                    created_at=self._clock,
                    updated_at=self._clock,
                )
                self._items[ref_id] = ref_item
            elif any(w in text for w in ("high", "identified", "found", "success", "vacuum full", "lock")):
                self._clock += 1
                ref_id = f"copper-refl-{self._clock}"
                ref_item = COPPERItem(
                    memory_id=ref_id,
                    task_id=proposal.task_id,
                    title=f"[Reflect: Reinforce] {proposal.title[:30]}",
                    raw_value=f"Counterfactual Lesson from {proposal.agent_id}: Positive diagnostic signal detected. Corroborate this hypothesis.",
                    source_agent=proposal.agent_id,
                    step_index=proposal.step_index,
                    is_reflection=True,
                    active=True,
                    created_at=self._clock,
                    updated_at=self._clock,
                )
                self._items[ref_id] = ref_item

            if self.trace is not None:
                self.trace.log(
                    "copper_operation",
                    memory_id=item_id,
                    task_id=proposal.task_id,
                    agent_id=proposal.agent_id,
                    proposal=proposal.__dict__,
                )
            return base_item

    def visible_keys(
        self,
        reader_id: str,
        task_id: str,
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryCard]:
        del reader_id
        with self._lock:
            active = [it for it in self._items.values() if it.task_id == task_id and it.active]
            if not active:
                return []

            q_tok = _tokens(query) if query else set()
            scored = []
            for it in active:
                overlap = len(q_tok & _tokens(it.title + " " + it.raw_value)) if q_tok else 1
                # COPPER boosts counterfactual reflections to prevent team repeated mistakes
                refl_bonus = 2.5 if it.is_reflection else 1.0
                score = overlap * refl_bonus + (it.updated_at / (self._clock + 1)) * 0.4
                scored.append((score, it))

            scored.sort(key=lambda s: s[0], reverse=True)
            ranked = [it for _, it in scored[:top_k]]

            return [
                MemoryCard(
                    memory_id=it.memory_id,
                    task_id=it.task_id,
                    title=it.title,
                    visibility="global",
                )
                for it in ranked
            ]

    def read(self, memory_id: str, reader_id: str, task_id: str) -> COPPERItem:
        with self._lock:
            item = self._items[memory_id]
            if self.trace is not None:
                self.trace.log_read(memory_id, reader_id=reader_id, task_id=task_id)
            return item
