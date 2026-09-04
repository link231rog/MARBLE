from __future__ import annotations

from typing import List, Optional

from .governed_memory import GovernedMemory
from .schema import MemoryCard, MemoryProposal, classify_topics


import os
import re
from typing import Tuple


def distill_proposal_output(
    output: str,
    max_title_words: int = 24,
    worker_model: Optional[str] = None,
) -> tuple[str, str]:
    """Pure LLM-driven Memory Condenser.

    Generates a high-quality, concise semantic summary key using the LLM,
    with a graceful sentence-based fallback for offline/test environments.
    """
    output = output.strip()
    if not output:
        return "Empty observation", ""

    # Strip wrapper prefixes
    clean = re.sub(
        r"^(?:Result from the (?:model|function)|As (?:an? )?[a-zA-Z0-9_]+):\s*",
        "",
        output,
        flags=re.IGNORECASE,
    ).strip()

    # 1. Pure LLM Summary Key Generation (skipped during tests/offline mode)
    model = worker_model or os.environ.get("MARBLE_WORKER_MODEL")
    if model and not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("MARBLE_OFFLINE_DISTILL")):
        try:
            from marble.llms.model_prompting import model_prompting

            prompt = (
                "Summarize the core diagnostic or research finding below into a single, concise "
                f"title key (maximum {max_title_words} words). "
                'Return ONLY a JSON object: {"summary_key": "<concise summary key>"}.\n'
                f"Finding:\n{clean[:1200]}"
            )
            resp = model_prompting(
                llm_model=model,
                messages=[{"role": "user", "content": prompt}],
                return_num=1,
                max_token_num=256,
                temperature=0.0,
            )[0]
            raw_text = str(getattr(resp, "content", "") or "")
            match = re.search(r'"summary_key"\s*:\s*"([^"]+)"', raw_text)
            if match:
                title = match.group(1).strip()
                words = title.split()
                if len(words) > max_title_words:
                    title = " ".join(words[:max_title_words])
                return title, clean[:800]
        except Exception:
            pass

    # 2. Deterministic Fallback (Offline / unit tests)
    finding_match = re.search(
        r"\[(?:Finding|Summary|Conclusion)\]:\s*([^\n]+)", clean, re.IGNORECASE
    )
    if finding_match:
        words = finding_match.group(1).strip().split()
        return " ".join(words[:max_title_words]), clean[:800]

    sentences = [s.strip() for s in re.split(r"[.\n]+", clean) if len(s.strip()) > 5]
    if sentences:
        words = sentences[0].split()
        return " ".join(words[:max_title_words]), clean[:800]

    words = clean.split()
    return " ".join(words[:max_title_words]), clean[:500]


def make_title(output: str, max_words: int = 24) -> str:
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
