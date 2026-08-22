from __future__ import annotations

from typing import Dict, List, Optional

from marble.memory import GovernedMemory
from marble.memory.adapter import MemoryAwareAgentAdapter


class MockAgent:
    """Scripted agent for Stage-0 smoke runs; no external API calls."""

    def __init__(self, agent_id: str, outputs: List[str]) -> None:
        self.agent_id = agent_id
        self.outputs = list(outputs)
        self.seen_cards: List[List] = []

    def act(self, task: str) -> str:
        return self.outputs.pop(0)


def run_mock_episode(
    memory: GovernedMemory,
    task_id: str,
    agent_outputs: Dict[str, List[str]],
    read_plan: Optional[Dict[str, List[str]]] = None,
) -> Dict:
    """Run one scripted episode through the adapter and return stats."""
    read_plan = read_plan or {}
    stats = {
        "task_id": task_id,
        "proposals_stored": 0,
        "visible_key_events": 0,
        "reads": [],
    }

    for agent_id, outputs in agent_outputs.items():
        agent = MockAgent(agent_id, outputs)
        adapter = MemoryAwareAgentAdapter(agent, memory)
        for _ in outputs:
            cards = adapter.before_step(task_id, agent_id)
            stats["visible_key_events"] += len(cards)
            output = agent.act(task_id)
            stored_id = adapter.after_step(output)
            if stored_id is not None:
                stats["proposals_stored"] += 1
            for memory_id in read_plan.get(agent_id, []):
                adapter.read(memory_id)
                stats["reads"].append(
                    {"reader_id": agent_id, "memory_id": memory_id}
                )

    return stats
