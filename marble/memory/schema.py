from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Visibility = Literal["private", "global", "absent"]


@dataclass(frozen=True)
class MemoryProposal:
    proposal_id: str
    task_id: str
    agent_id: str
    source: str
    title: str
    raw_value: str
    step_index: int
    topics: tuple = ()  # fixed-taxonomy tags, soft input only (schema-and-reward.md)


@dataclass(frozen=True)
class MemoryTargetState:
    exists: bool
    visibility: Visibility
    owner_id: Optional[str] = None
    supersedes: Optional[str] = None

    def __post_init__(self) -> None:
        if self.visibility not in ("private", "global", "absent"):
            raise ValueError("visibility must be private, global, or absent")
        if self.visibility == "absent" and self.exists:
            raise ValueError("absent target state cannot exist")
        if self.visibility != "absent" and not self.exists:
            raise ValueError("non-absent target state must exist")
        if self.visibility == "private" and not self.owner_id:
            raise ValueError("private target state requires owner_id")
        if self.visibility == "global" and self.owner_id is not None:
            raise ValueError("global target state cannot have owner_id")


@dataclass(frozen=True)
class MemoryCard:
    memory_id: str
    task_id: str
    title: str
    visibility: Literal["private", "global"]
    owner_id: Optional[str]


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    proposal_id: str
    task_id: str
    title: str
    raw_value: str
    visibility: Literal["private", "global"]
    owner_id: Optional[str]
    source_agent: str
    source: str
    step_index: int
    active: bool
    supersedes: Optional[str]
    created_at: int

    def card(self) -> MemoryCard:
        return MemoryCard(
            memory_id=self.memory_id,
            task_id=self.task_id,
            title=self.title,
            visibility=self.visibility,
            owner_id=self.owner_id,
        )
