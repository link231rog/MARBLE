"""Bridge between the real MARBLE Engine/BaseAgent and GovernedMemory (spec §6-§7).

Module level imports NOTHING from marble.engine / marble.agent / marble.llms,
so offline tests stay litellm-free; heavy imports happen inside factories.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from marble.memory.adapter import distill_proposal_output, make_title
from marble.memory.governed_memory import GovernedMemory
from marble.memory.rewards import token_count
from marble.memory.schema import MemoryProposal, classify_topics


class MemoryStep:
    """Per-agent act() lifecycle around a shared GovernedMemory (spec §6.2)."""

    def __init__(
        self,
        memory: Optional[GovernedMemory],
        selector_fn: Optional[Callable[[str], str]] = None,
        max_cards: int = 6,
        max_reads_per_step: int = 2,
        selector: str = "callable",
        task_goal: str = "",
        agent_role_map: Optional[Dict[str, str]] = None,
        baseline: str = "heuristic",
        worker_model: Optional[str] = None,
    ):
        self.memory = memory
        # selector: "callable" (use selector_fn) or "top" (rank-order, no API)
        self.selector = selector
        self.selector_fn = selector_fn
        self.max_cards = max_cards
        self.max_reads_per_step = max_reads_per_step
        self.task_id: str = ""
        self.steps: Dict[str, int] = {}
        self.reads_this_episode = 0
        self.selection_rejections: List[Dict[str, Any]] = []
        self.task_goal = task_goal
        self.agent_role_map = dict(agent_role_map or {})
        self.baseline = baseline
        self.worker_model = worker_model
        self._context_by_agent: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------ before
    def before_act(self, agent_id: str, task_text: str) -> str:
        self.steps[agent_id] = self.steps.get(agent_id, 0) + 1
        if self.memory is None:
            return task_text
        controller = getattr(self.memory, "controller", None)
        if controller is not None and hasattr(controller, "set_context"):
            controller.set_context(self.task_goal, self.agent_role_map)
        eff_top_k = 20 if self.baseline == "global_add_all" else self.max_cards
        cards = self.memory.visible_keys(
            reader_id=agent_id, task_id=self.task_id, query=task_text, top_k=eff_top_k
        )
        trace = getattr(self.memory, "trace", None)
        if trace is not None and hasattr(trace, "log_exposure"):
            trace.log_exposure(
                [card.memory_id for card in cards],
                reader_id=agent_id,
                task_id=self.task_id,
            )
        key_lines: List[str] = []
        notes: List[str] = []
        if cards:
            key_lines = [
                "Shared memory keys:",
                *[f"- [M{i+1}] {c.title} ({c.visibility})" for i, c in enumerate(cards)],
            ]
            notes = self._read_selected(agent_id, cards)
            if hasattr(self.memory, "distill"):
                notes = self.memory.distill(task_text, notes)
            print(
                f"[Memory] [{self.task_id}][{agent_id}] exposed {len(cards)} cards | read {len(notes)} raw items",
                flush=True,
            )
        note_lines = ["Read notes:", *notes] if notes else []
        parts = [task_text, *key_lines, *note_lines]
        augmented = "\n".join(parts)
        self._context_by_agent[agent_id] = {
            "memory_card_tokens": token_count("\n".join(key_lines)),
            "injected_memory_tokens": token_count("\n".join(notes)),
        }
        return augmented

    def _read_selected(self, agent_id: str, cards) -> List[str]:
        assert self.memory is not None  # only called from before_act after the None guard
        max_reads = len(cards) if self.baseline == "global_add_all" else self.max_reads_per_step
        chosen = self._select_ids(agent_id, cards)[: max_reads]
        notes: List[str] = []
        for mid in chosen:
            try:
                value = self.memory.read(mid, reader_id=agent_id, task_id=self.task_id)
            except (KeyError, PermissionError):
                continue
            notes.append(f"- [{value.title}] {value.raw_value}")
            self.reads_this_episode += 1
        return notes

    def _select_ids(self, agent_id: str, cards) -> List[str]:
        if self.selector == "top":
            # rank-order default: read the top-ranked cards without an extra API call
            max_reads = len(cards) if self.baseline == "global_add_all" else self.max_reads_per_step
            return [c.memory_id for c in cards[: max_reads]]
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
        controller = getattr(self.memory, "controller", None)
        if controller is not None and hasattr(controller, "set_context"):
            controller.set_context(self.task_goal, self.agent_role_map)
        title, condensed_value = distill_proposal_output(output, worker_model=self.worker_model)
        proposal = MemoryProposal(
            proposal_id=f"{self.task_id}:{agent_id}:{self.steps.get(agent_id, 0)}",
            task_id=self.task_id,
            agent_id=agent_id,
            source="worker",
            title=title,
            raw_value=condensed_value,
            step_index=self.steps.get(agent_id, 0),
            topics=classify_topics(output),
        )
        metadata: Dict[str, Any] = dict(self._context_by_agent.get(agent_id, {}))
        if controller is not None:
            metadata.update({
                "controller_prompt": getattr(controller, "last_prompt", ""),
                "controller_output": getattr(controller, "last_raw", ""),
                "active_memory_index": [
                    it.__dict__ for it in self.memory.bank.all_items()
                    if it.active and it.task_id == proposal.task_id
                ],
            })
        item = self.memory.submit(proposal, **metadata)
        if item is not None:
            print(
                f"[Memory] [{self.task_id}][{agent_id}] proposal -> stored as {item.visibility} (id: {item.memory_id})",
                flush=True,
            )
        else:
            print(
                f"[Memory] [{self.task_id}][{agent_id}] proposal -> absent/rejected",
                flush=True,
            )
        return item.memory_id if item is not None else None


def check_consensus(environment_name: str, agents_results: List[Dict[str, Any]]) -> bool:
    """Detect consensus early-exit to avoid redundant fixed-loop iterations."""
    if not agents_results or len(agents_results) < 3:
        return False
    import re
    if "DB" in str(environment_name):
        candidates = [
            "INSERT_LARGE_DATA", "MISSING_INDEXES", "LOCK_CONTENTION",
            "VACUUM", "REDUNDANT_INDEX", "FETCH_LARGE_DATA",
            "POOR_JOIN_PERFORMANCE", "CPU_CONTENTION"
        ]
        votes: Dict[str, int] = {}
        for r_dict in agents_results:
            text = str(list(r_dict.values())[0] if r_dict else "")
            for c in candidates:
                if re.search(rf"\b(cause|conclude|concluded|identified|bottleneck|anomaly)\b.*?\b{c}\b", text, re.IGNORECASE) or \
                   re.search(rf"\b{c}\b.*?\b(is the root cause|is the cause|identified)\b", text, re.IGNORECASE):
                    votes[c] = votes.get(c, 0) + 1
        if any(cnt >= 3 for cnt in votes.values()):
            return True
    elif "Research" in str(environment_name):
        full_text = " ".join(str(list(r.values())[0]) for r in agents_results if r)
        if all(f"[Question {i}]" in full_text for i in range(1, 6)) or \
           all(f"Question {i}:" in full_text for i in range(1, 6)):
            return True
    return False


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
    """GovernedEngine = Engine injecting GovernedAgent on every seat with consensus check."""
    if engine_cls is None:
        from marble.engine.engine import Engine as engine_cls
    if agent_cls is None:
        agent_cls = build_governed_agent_cls()

    class GovernedEngine(engine_cls):
        memory_harness: Optional[MemoryStep] = None

        def start(self):
            if hasattr(self, "planner") and hasattr(self.planner, "decide_next_step"):
                orig_decide = self.planner.decide_next_step
                def hooked_decide(agents_results):
                    env_name = getattr(self.environment, "name", "")
                    if check_consensus(env_name, agents_results):
                        self.logger.info(f"[Consensus] Early exit triggered: consensus reached in {env_name}.")
                        print(f"[Engine] >>> Early exit triggered: consensus reached across agents!", flush=True)
                        return False
                    return orig_decide(agents_results)
                self.planner.decide_next_step = hooked_decide

            if hasattr(self, "planner") and hasattr(self.planner, "summarize_output"):
                orig_summarize = self.planner.summarize_output
                def hooked_summarize(summary_text, task_text, output_format_text):
                    env_name = getattr(self.environment, "name", "")
                    if env_name == "DB Environment":
                        output_format_text = (
                            str(output_format_text or "")
                            + "\nCRITICAL FINAL DECISION: You MUST conclude with the definitive root cause decision in JSON format: {\"root_causes\": [\"<CAUSE1>\", \"<CAUSE2>\"]}. Do NOT output next steps or process suggestions."
                        )
                    elif env_name == "Research Environment":
                        output_format_text = (
                            str(output_format_text or "")
                            + "\nCRITICAL FINAL DECISION: Compile the definitive 5-Question (5q) research proposal addressing Question 1 through Question 5 in complete detail."
                        )
                    return orig_summarize(summary_text, task_text, output_format_text)
                self.planner.summarize_output = hooked_summarize

            return super().start()

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
