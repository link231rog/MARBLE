"""Bridge between the real MARBLE Engine/BaseAgent and GovernedMemory (spec §6-§7).

Module level imports NOTHING from marble.engine / marble.agent / marble.llms,
so offline tests stay litellm-free; heavy imports happen inside factories.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from marble.memory.adapter import make_title
from marble.memory.governed_memory import GovernedMemory
from marble.memory.schema import MemoryProposal


class MemoryStep:
    """Per-agent act() lifecycle around a shared GovernedMemory (spec §6.2)."""

    def __init__(
        self,
        memory: Optional[GovernedMemory],
        selector_fn: Optional[Callable[[str], str]] = None,
        max_cards: int = 6,
        max_reads_per_step: int = 2,
    ):
        self.memory = memory
        self.selector_fn = selector_fn
        self.max_cards = max_cards
        self.max_reads_per_step = max_reads_per_step
        self.task_id: str = ""
        self.steps: Dict[str, int] = {}
        self.reads_this_episode = 0
        self.selection_rejections: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ before
    def before_act(self, agent_id: str, task_text: str) -> str:
        self.steps[agent_id] = self.steps.get(agent_id, 0) + 1
        if self.memory is None:
            return task_text
        cards = self.memory.visible_keys(
            reader_id=agent_id, task_id=self.task_id, query=task_text, top_k=self.max_cards
        )
        parts = [task_text]
        if cards:
            parts.append("Shared memory keys:")
            parts += [f"- {c.memory_id} | {c.title} | {c.visibility} | owner={c.owner_id}" for c in cards]
            notes = self._read_selected(agent_id, cards)
            if notes:
                parts.append("Read notes:")
                parts += notes
        return "\n".join(parts)

    def _read_selected(self, agent_id: str, cards) -> List[str]:
        assert self.memory is not None  # only called from before_act after the None guard
        chosen = self._select_ids(agent_id, cards)[: self.max_reads_per_step]
        notes: List[str] = []
        for mid in chosen:
            try:
                value = self.memory.read(mid, reader_id=agent_id, task_id=self.task_id)
            except (KeyError, PermissionError):
                continue
            notes.append(f"- [{mid}] {value}")
            self.reads_this_episode += 1
        return notes

    def _select_ids(self, agent_id: str, cards) -> List[str]:
        if self.selector_fn is None:
            return []
        listing = "\n".join(f"- {c.memory_id} | {c.title}" for c in cards)
        prompt = (
            f"task: {self.task_id}\nkey memories:\n{listing}\n"
            f'Select up to {self.max_reads_per_step} memory_ids worth reading. '
            'Respond with ONLY JSON: {"memory_ids": ["..."]}'
        )
        try:
            parsed = json.loads(self.selector_fn(prompt))
            ids = parsed["memory_ids"]
            if not isinstance(ids, list):
                raise ValueError("memory_ids must be a list")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            self.selection_rejections.append({"agent_id": agent_id, "reason": str(exc)})
            return []
        known = {c.memory_id for c in cards}
        return [mid for mid in ids if isinstance(mid, str) and mid in known]

    # ------------------------------------------------------------------- after
    def after_act(self, agent_id: str, output: str) -> Optional[str]:
        if self.memory is None or not output.strip():
            return None
        proposal = MemoryProposal(
            proposal_id=f"{self.task_id}:{agent_id}:{self.steps.get(agent_id, 0)}",
            task_id=self.task_id,
            agent_id=agent_id,
            source="worker",
            title=make_title(output),
            raw_value=output,
            step_index=self.steps.get(agent_id, 0),
        )
        item = self.memory.submit(proposal)
        return item.memory_id if item is not None else None


def build_governed_agent_cls(base_agent_cls: Any = None):
    """GovernedAgent = BaseAgent whose act() runs the governed lifecycle."""
    if base_agent_cls is None:
        from marble.agent.base_agent import BaseAgent as base_agent_cls

    class GovernedAgent(base_agent_cls):
        governed: Optional[MemoryStep] = None

        def act(self, task):
            harness = getattr(self, "governed", None)
            augmented = harness.before_act(self.agent_id, task) if harness else task
            result = super().act(augmented)
            if harness:
                harness.after_act(self.agent_id, result[0])
            return result

    return GovernedAgent


def build_governed_engine_cls(engine_cls: Any = None, agent_cls: Any = None):
    """GovernedEngine = Engine injecting GovernedAgent on every seat."""
    if engine_cls is None:
        from marble.engine.engine import Engine as engine_cls
    if agent_cls is None:
        agent_cls = build_governed_agent_cls()

    class GovernedEngine(engine_cls):
        memory_harness: Optional[MemoryStep] = None

        def _initialize_agents(self, agent_configs):
            agents = []
            for agent_config in agent_configs:
                agent_llm = agent_config.get("llm", self.config.llm)
                agent = agent_cls(config=agent_config, env=self.environment, model=agent_llm)
                agent.governed = self.memory_harness
                agents.append(agent)
                # ponytail: duck-typed minecraft registration; full isinstance needs the env import chain
                if "agent_port" in agent_config and hasattr(self.environment, "register_agent"):
                    self.environment.register_agent(
                        agent_config.get("agent_id"), agent_config.get("agent_port")
                    )
            return agents

    return GovernedEngine
