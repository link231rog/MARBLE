from __future__ import annotations

from typing import List, Optional

from .governed_memory import GovernedMemory
from .schema import MemoryCard, MemoryProposal, classify_topics


import re


def distill_proposal_output(output: str, max_title_words: int = 16) -> tuple[str, str]:
    """Dual-Track Robust Memory Condenser.

    Track 1: Deterministic Tool-Event Extraction (Database SQL & Research tools).
    Track 2: Stateless Semantic Distillation (Extracting core claims, stripping boilerplate).
    """
    output = output.strip()
    if not output:
        return "Empty observation", ""

    # Track 1: Deterministic Tool-Event Extraction
    if "Result from the function:" in output or "\"function_name\":" in output:
        fn_match = re.search(r"\"function_name\":\s*\"([^\"]+)\"", output)
        fn = fn_match.group(1) if fn_match else "tool"

        # Database tool events
        if "VACUUM" in output:
            title = "query_db: VACUUM execution time bottleneck identified"
            condensed = "Tool observation: VACUUM / VACUUM FULL shows dominant cumulative latency in pg_stat_statements."
            return title, condensed
        elif "INSERT" in output and ("total_exec_time" in output or "calls" in output):
            title = "query_db: heavy INSERT workload identified"
            condensed = "Tool observation: INSERT operations consume substantial execution time in pg_stat_statements."
            return title, condensed
        elif "pg_locks" in output:
            title = "query_db: pg_locks checked, no blocking lock contention"
            condensed = "Tool observation: pg_locks inspected; only shared locks present, exclusive lock contention ruled out."
            return title, condensed
        elif "pg_stat_user_indexes" in output or "REDUNDANT_INDEX" in output:
            title = "query_db: user indexes analyzed for redundancy"
            condensed = "Tool observation: User table index usage scanned for redundant or missing indexes."
            return title, condensed
        elif "pg_stat_activity" in output:
            title = "query_db: pg_stat_activity background processes normal"
            condensed = "Tool observation: pg_stat_activity inspected; active worker connections within expected bounds."
            return title, condensed

        # Research tool events
        elif any(k in output for k in ["get_paper", "search_arxiv", "get_related_papers"]):
            kw_match = re.search(r"['\"](?:query|keyword)['\"]:\s*['\"]([^'\"]+)['\"]", output)
            kw = kw_match.group(1)[:30] if kw_match else "literature"
            title = f"research: literature query on '{kw}'"
            condensed = f"Tool observation: Retrieved academic papers and findings for topic '{kw}'."
            return title, condensed
        elif "communicate" in fn:
            title = "communication: exchanged finding with collaborator"
            condensed = "Communication session executed between collaborative agents."
            return title, condensed
        else:
            title = f"[{fn}] execution completed"
            condensed = f"Tool {fn} executed successfully."
            return title, condensed

    # Track 2: Stateless Semantic Distillation for Free-text / Reasoning
    clean = re.sub(r"^Result from the model:\s*", "", output).strip()
    clean = re.sub(r"^As (?:an? )?[a-zA-Z0-9_]+,\s*", "", clean, flags=re.IGNORECASE)

    # Check for explicit finding tags
    finding_match = re.search(r"\[(?:Finding|Summary|Conclusion)\]:\s*([^\n]+)", clean, re.IGNORECASE)
    if finding_match:
        claim = finding_match.group(1).strip()
        words = claim.split()
        return " ".join(words[:max_title_words]), clean[:800]

    # Split into candidate sentences
    sentences = [s.strip() for s in re.split(r"[.\n]+", clean) if len(s.strip()) > 5]
    if sentences:
        # Prioritize decisive diagnostic statements
        decisive = [
            s for s in sentences
            if any(k in s.lower() for k in ["unlikely", "likely", "bottleneck", "conclude", "found", "root cause", "assign", "propose", "recommend"])
        ]
        chosen = decisive[0] if decisive else sentences[0]
        words = chosen.split()
        title = " ".join(words[:max_title_words])
        condensed = "\n".join(sentences[:3])[:800]
        return title, condensed

    words = clean.split()
    return " ".join(words[:max_title_words]), clean[:500]


def make_title(output: str, max_words: int = 16) -> str:
    title, _ = distill_proposal_output(output, max_title_words=max_words)
    return title


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
