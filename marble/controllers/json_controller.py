"""LLM-backed controller emitting STRICT JSON decisions (spec §9.3).

Invalid outputs are rejected AND recorded in ``rejections`` — never silently
coerced into a valid guess.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from marble.memory.schema import (
    MemoryItem,
    MemoryProposal,
    MemoryTargetState,
    TOPIC_TAXONOMY,
)

VALID_VISIBILITIES = ("absent", "private", "global")

# Dynamic input-field ablations accepted in drop_fields.
_BLOCK_DROPS = frozenset({"topics", "memory_summary", "active_memory_index"})


class JsonController:
    """decide() via llm_fn(prompt) -> str; strict-JSON contract enforced here.

    Prompt follows fixed system context plus TASK/PROPOSAL/ACTIVE MEMORY INDEX.
    Agent roles remain static; only agent_reference appears in dynamic input.
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        max_value_chars: int = 512,
        drop_fields: tuple = (),
        agent_capabilities: tuple = (),
        task_goal: str = "",
        agent_role_map: Optional[Mapping[str, str]] = None,
    ):
        self.llm_fn = llm_fn
        self.max_value_chars = max_value_chars
        self.drop_fields = frozenset(drop_fields)
        self.agent_role_map = dict(agent_role_map or {
            str(index): value for index, value in enumerate(agent_capabilities)
        })
        self.task_goal = task_goal
        self.rejections: List[Dict[str, Any]] = []
        self.last_prompt = ""
        self.last_raw = ""

    def set_context(
        self,
        task_goal: str = "",
        agent_role_map: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Update per-episode prompt context without changing the output contract."""
        self.task_goal = task_goal
        if agent_role_map is not None:
            self.agent_role_map = dict(agent_role_map)

    # ------------------------------------------------------------------ prompt
    def build_prompt(self, proposal: MemoryProposal, current_state: Sequence[MemoryItem]) -> str:
        same_title = _find_same_title(proposal, current_state)
        lines: List[str] = [
            "[SYSTEM]",
            f"agent_role_map: {json.dumps(self.agent_role_map, ensure_ascii=False, sort_keys=True)}",
            f"topic_taxonomy: {list(TOPIC_TAXONOMY)}",
            "visibility: absent removes memory; private exposes only to owner; global exposes to all agents.",
            "supersedes: use an exact active memory_id only when replacing that memory.",
            'output: ONLY {"visibility": "absent"|"private"|"global", "supersedes": null}.',
        ]

        # [TASK] — task context, present in all variants
        lines.append("[TASK]")
        lines.append(f"task_goal: {self.task_goal or '(unspecified)'}")

        # [PROPOSAL]
        lines.append("[PROPOSAL]")
        lines.append(f"agent_reference: {proposal.agent_id}")
        if "title" not in self.drop_fields:
            lines.append(f"title: {proposal.title}")
        if "value" not in self.drop_fields:
            lines.append(f"value: {proposal.raw_value[: self.max_value_chars]}")
        if "source" not in self.drop_fields:
            lines.append(f"source: {proposal.source}")
        if "topics" not in self.drop_fields and proposal.topics:
            lines.append(f"topics: {list(proposal.topics)}")

        # [ACTIVE MEMORY INDEX]
        if "active_memory_index" not in self.drop_fields:
            lines.append("[ACTIVE MEMORY INDEX]")
            for it in current_state:
                if not (it.active and it.task_id == proposal.task_id):
                    continue
                summary = "" if "memory_summary" in self.drop_fields else f" | summary: {it.summary}"
                topics = "" if "topics" in self.drop_fields else f" | topics: {list(it.topics)}"
                lines.append(
                    f"- memory_id: {it.memory_id} | title: {it.title}{summary}{topics} | "
                    f"visibility: {it.visibility} | owner_id: {it.owner_id}"
                )
            lines.append(f"same_title_active: {'true' if same_title else 'false'}")
        else:
            lines.append("same_title_active: 'false' (active memory hidden)")

        return "\n".join(lines)

    # ------------------------------------------------------------------ decide
    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        self.last_prompt = self.build_prompt(proposal, current_state)
        raw = self.llm_fn(self.last_prompt)
        self.last_raw = raw
        error = self._validate(raw, current_state, proposal)
        if error is not None:
            self.rejections.append(
                {"proposal_id": proposal.proposal_id, "raw": raw, "reason": error}
            )
            return MemoryTargetState(exists=False, visibility="absent")
        parsed = json.loads(raw)
        visibility = parsed["visibility"]
        if visibility == "absent":
            return MemoryTargetState(exists=False, visibility="absent")
        supersedes = parsed.get("supersedes")
        # input ablation: active memory hidden -> controller may not supersede
        if "active_memory_index" in self.drop_fields:
            supersedes = None
        resolved = _resolve_supersedes(supersedes, current_state, proposal)
        return MemoryTargetState(
            exists=True,
            visibility=visibility,
            owner_id=proposal.agent_id if visibility == "private" else None,
            supersedes=resolved,
        )

    # --------------------------------------------------------------- validation
    def _validate(
        self,
        raw: str,
        current_state: Sequence[MemoryItem],
        proposal: MemoryProposal,
    ) -> Optional[str]:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return f"not valid JSON: {exc}"
        if not isinstance(parsed, dict):
            return "not a JSON object"
        if set(parsed) != {"visibility", "supersedes"}:
            return f"expected exactly keys visibility/supersedes, got {sorted(parsed)}"
        visibility = parsed["visibility"]
        if visibility not in VALID_VISIBILITIES:
            return f"invalid visibility {visibility!r}"
        supersedes = parsed["supersedes"]
        if visibility == "absent":
            if supersedes is not None:
                return "absent decision must have null supersedes"
            return None
        if supersedes is not None and _resolve_supersedes(supersedes, current_state, proposal) is None:
            return f"supersedes target not found among active items: {supersedes!r}"
        return None


def _find_same_title(proposal: MemoryProposal, current_state: Sequence[MemoryItem]) -> bool:
    normalized = proposal.title.strip().casefold()
    return any(
        it.active
        and it.task_id == proposal.task_id
        and it.title.strip().casefold() == normalized
        for it in current_state
    )


def _resolve_supersedes(
    supersedes: Any,
    current_state: Sequence[MemoryItem],
    proposal: MemoryProposal,
) -> Optional[str]:
    if supersedes is None:
        return None
    wanted = str(supersedes).strip().casefold()
    for it in current_state:
        if not (it.active and it.task_id == proposal.task_id):
            continue
        if it.memory_id == supersedes or str(supersedes) == it.memory_id:
            return it.memory_id
    return None
