from __future__ import annotations

from typing import List, Optional

from .governed_memory import GovernedMemory
from .schema import MemoryCard, MemoryProposal, classify_topics


def make_title(output: str, max_words: int = 16) -> str:
    words = output.strip().split()
    return " ".join(words[:max_words])


class MemoryAwareAgentAdapter:
    """Minimal wrapper adding governed memory around an agent's act() step."""

    def __init__(self, agent, memory: GovernedMemory, source: str = "worker") -> None:
        self.agent = agent
        self.memory = memory
        self.source = source
        self._task_id: Optional[str] = None
        self._agent_id: Optional[str] = None
        self._step_index = 0

    def before_step(
        self,
        task_id: str,
        agent_id: str,
        query: Optional[str] = None,
        top_k: int = 6,
    ) -> List[MemoryCard]:
        self._task_id = task_id
        self._agent_id = agent_id
        self._step_index += 1
        return self.memory.visible_keys(agent_id, task_id, query=query, top_k=top_k)

    def after_step(self, output: str) -> Optional[str]:
        if not self._task_id or not self._agent_id:
            raise RuntimeError("before_step must run before after_step")
        proposal = MemoryProposal(
            proposal_id=f"{self._task_id}:{self._agent_id}:{self._step_index}",
            task_id=self._task_id,
            agent_id=self._agent_id,
            source=self.source,
            title=make_title(output),
            raw_value=output,
            step_index=self._step_index,
            topics=classify_topics(output),
        )
        item = self.memory.submit(proposal)
        return item.memory_id if item is not None else None

    def read(self, memory_id: str) -> str:
        if not self._task_id or not self._agent_id:
            raise RuntimeError("before_step must run before read")
        return self.memory.read(memory_id, self._agent_id, self._task_id).raw_value


def format_cards(cards: List[MemoryCard]) -> str:
    lines = [
        f"- [{card.visibility}] {card.memory_id}: {card.title}"
        for card in cards
    ]
    return "\n".join(lines) if lines else "(no memories)"
