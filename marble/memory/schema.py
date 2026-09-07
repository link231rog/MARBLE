from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional


Visibility = Literal["absent", "global", "targeted"]

# Fixed initial topic taxonomy (schema-and-reward.md). Soft input only.
TOPIC_TAXONOMY = (
    "code", "research", "retrieval", "database",
    "testing", "planning", "analysis",
)

_TOPIC_RE = {t: re.compile(rf"\b{re.escape(t)}\b", re.I) for t in TOPIC_TAXONOMY}


def classify_topics(text: str) -> tuple:
    """Tag text with fixed-taxonomy topics (word-boundary match)."""
    text = text or ""
    return tuple(t for t, rx in _TOPIC_RE.items() if rx.search(text))


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
    target_recipients: tuple = ()  # authorized recipient agent IDs
    supersedes: Optional[str] = None

    def __post_init__(self) -> None:
        if self.visibility not in ("absent", "global", "targeted"):
            raise ValueError("visibility must be absent, global, or targeted")
        if self.visibility == "absent" and self.exists:
            raise ValueError("absent target state cannot exist")
        if self.visibility != "absent" and not self.exists:
            raise ValueError("non-absent target state must exist")
        if self.visibility == "targeted" and not self.target_recipients:
            raise ValueError("targeted target state requires target_recipients")

    @classmethod
    def targeted(
        cls,
        recipients: tuple | list,
        supersedes: Optional[str] = None,
    ) -> MemoryTargetState:
        return cls(
            exists=True,
            visibility="targeted",
            target_recipients=tuple(sorted(set(recipients))),
            supersedes=supersedes,
        )


@dataclass(frozen=True)
class MemoryCard:
    memory_id: str
    task_id: str
    title: str
    visibility: Literal["global", "targeted"] = "global"
    target_recipients: tuple = ()


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    proposal_id: str
    task_id: str
    title: str
    raw_value: str
    visibility: Literal["global", "targeted"]
    source_agent: str
    source: str
    step_index: int
    active: bool
    supersedes: Optional[str]
    created_at: int
    summary: str = ""
    topics: tuple = ()
    target_recipients: tuple = ()

    def card(self) -> MemoryCard:
        return MemoryCard(
            memory_id=self.memory_id,
            task_id=self.task_id,
            title=self.title,
            visibility=self.visibility,
            target_recipients=self.target_recipients,
        )
