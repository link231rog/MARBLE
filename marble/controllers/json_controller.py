"""LLM-backed controller emitting STRICT JSON decisions (spec §9.3).

Invalid outputs are rejected AND recorded in ``rejections`` — never silently
coerced into a valid guess.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from marble.memory.schema import (
    TOPIC_TAXONOMY,
    MemoryItem,
    MemoryProposal,
    MemoryTargetState,
)

VALID_VISIBILITIES = ("absent", "global")

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
        self.last_parse_status = ""
        self.last_log_prob: Optional[float] = None

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
        agent_ids = sorted(self.agent_role_map.keys())
        example_agent_1 = f'["{agent_ids[0]}"]' if agent_ids else '["agent_1"]'
        example_agent_2 = (
            f'["{agent_ids[0]}", "{agent_ids[1]}"]'
            if len(agent_ids) >= 2
            else '["agent_1", "agent_2"]'
        )
        lines: List[str] = [
            "[SYSTEM]",
            f"agent_role_map: {json.dumps(self.agent_role_map, ensure_ascii=False, sort_keys=True)}",
            f"topic_taxonomy: {list(TOPIC_TAXONOMY)}",
            "visibility: absent removes memory; global exposes to all agents; a JSON array of agent IDs (e.g. "
            f'{example_agent_1} or {example_agent_2}) from agent_role_map routes strictly to those agents.',
            "update: use an exact active memory_id only when updating that memory.",
            'output: ONLY {"visibility": "absent"|"global"|["<agent_id>", ...], "update": null}.',
        ]

        # [TASK] — task context, present in all variants
        lines.append("[TASK]")
        if "task_goal" not in self.drop_fields:
            lines.append(f"task_goal: {self.task_goal or '(unspecified)'}")

        # [PROPOSAL]
        lines.append("[PROPOSAL]")
        if "agent_tag" not in self.drop_fields:
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
                recipients = list(it.target_recipients)
                lines.append(
                    f"- memory_id: {it.memory_id} | title: {it.title}{summary}{topics} | "
                    f"visibility: {it.visibility} | target_recipients: {recipients}"
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
        self.last_log_prob = getattr(self.llm_fn, "last_log_prob", None)
        error = self._validate(raw, current_state, proposal)
        if error is not None:
            self.last_parse_status = (
                "format_error"
                if ("not valid JSON" in error or "not a JSON object" in error)
                else "schema_error"
            )
            self.rejections.append(
                {"proposal_id": proposal.proposal_id, "raw": raw, "reason": error}
            )
            return MemoryTargetState(exists=False, visibility="absent")
        self.last_parse_status = "valid_json"
        parsed = json.loads(raw)
        visibility = parsed["visibility"]
        if visibility == "absent":
            return MemoryTargetState(exists=False, visibility="absent")
        update_target = parsed.get("update") if "update" in parsed else parsed.get("supersedes")
        # input ablation: active memory hidden -> controller may not update
        if "active_memory_index" in self.drop_fields:
            update_target = None
        resolved = _resolve_supersedes(update_target, current_state, proposal)

        if visibility == "global":
            return MemoryTargetState(
                exists=True,
                visibility="global",
                target_recipients=(),
                supersedes=resolved,
                update=resolved,
            )
        elif isinstance(visibility, list):
            recipients = tuple(sorted(set(visibility)))
            return MemoryTargetState(
                exists=True,
                visibility="targeted",
                target_recipients=recipients,
                supersedes=resolved,
                update=resolved,
            )
        else:
            return MemoryTargetState(exists=False, visibility="absent")

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
        allowed_key_sets = ({"visibility", "update"}, {"visibility", "supersedes"})
        if set(parsed) not in allowed_key_sets:
            return f"expected exactly keys visibility/update, got {sorted(parsed)}"
        visibility = parsed["visibility"]
        if visibility not in ("absent", "global"):
            if not isinstance(visibility, list):
                return f"invalid visibility {visibility!r}"
            if len(visibility) == 0:
                return "recipient list cannot be empty"
            if "global" in visibility:
                return "'global' cannot be included in recipient list"
            known_agents = set(self.agent_role_map.keys())
            if known_agents:
                for aid in visibility:
                    if not isinstance(aid, str) or aid not in known_agents:
                        return f"unknown recipient agent_id {aid!r}"
            else:
                for aid in visibility:
                    if not isinstance(aid, str):
                        return f"recipient agent_id {aid!r} must be a string"
        update_target = parsed.get("update") if "update" in parsed else parsed.get("supersedes")
        if visibility == "absent":
            if update_target is not None:
                return "absent decision must have null update (null supersedes)"
            return None
        if update_target is not None and _resolve_supersedes(update_target, current_state, proposal) is None:
            return f"update target not found among active items: {update_target!r}"
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
    for it in current_state:
        if not (it.active and it.task_id == proposal.task_id):
            continue
        if it.memory_id == supersedes or str(supersedes) == it.memory_id:
            return it.memory_id
    return None
