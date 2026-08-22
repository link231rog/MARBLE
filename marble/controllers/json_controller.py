"""LLM-backed controller emitting STRICT JSON decisions (spec §9.3).

Invalid outputs are rejected AND recorded in ``rejections`` — never silently
coerced into a valid guess.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from marble.memory.schema import MemoryItem, MemoryProposal, MemoryTargetState

VALID_VISIBILITIES = ("absent", "private", "global")


class JsonController:
    """decide() via llm_fn(prompt) -> str; strict-JSON contract enforced here."""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        max_value_chars: int = 512,
        drop_fields: tuple = (),
    ):
        self.llm_fn = llm_fn
        self.max_value_chars = max_value_chars
        self.drop_fields = frozenset(drop_fields)
        self.rejections: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ prompt
    def build_prompt(self, proposal: MemoryProposal, current_state: Sequence[MemoryItem]) -> str:
        same_title = _find_same_title(proposal, current_state)
        fields = {
            "title": proposal.title,
            "value": proposal.raw_value[: self.max_value_chars],
            "source": proposal.source,
            "agent_id": proposal.agent_id,
            "task_id": proposal.task_id,
            "step_index": proposal.step_index,
        }
        lines = [
            "Decide whether this agent output should become shared team memory.",
        ]
        lines += [f"{name}: {val}" for name, val in fields.items() if name not in self.drop_fields]
        lines.append("active memories:")
        lines += [
            f"- {it.memory_id} | {it.title} | {it.visibility} | owner={it.owner_id}"
            for it in current_state
            if it.active and it.task_id == proposal.task_id
        ]
        lines.append(f"same_title_active: {'true' if same_title else 'false'}")
        lines.append(
            'Respond with ONLY this JSON: {"visibility": "absent"|"private"|"global", '
            '"supersedes": null} — replace supersedes with a listed memory_id/title '
            "only when this output updates that memory."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ decide
    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        raw = self.llm_fn(self.build_prompt(proposal, current_state))
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
        if it.memory_id == supersedes or str(supersedes) == it.memory_id or it.title.strip().casefold() == wanted:
            return it.memory_id
    return None
