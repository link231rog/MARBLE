"""Linear-policy controller trainable via offline SFT (spec Stage 3)."""

from __future__ import annotations

import itertools
import json
import re
from typing import Any, Dict, List, Sequence

from marble.controllers.heuristic import _find_supersedes
from marble.memory.schema import MemoryItem, MemoryProposal, MemoryTargetState

VISIBILITIES = ("absent", "targeted", "global")

_SHARED_SIGNALS = re.compile(
    r"\b(share|shared|all|team|decision|result|consensus|plan|finding|solution|conclusion|agreed|summary)\b", re.IGNORECASE
)
_PRIVATE_SIGNALS = re.compile(
    r"\b(private|draft|scratch|internal|self|local|note|todo|working|my|mine|step|investigating)\b", re.IGNORECASE
)
_ACTION_SIGNALS = re.compile(
    r"\b(select|insert|update|create|table|query|traceback|error|psql|stdout|stderr)\b", re.IGNORECASE
)


def canonical_action(visibility: Any, target_recipients: Any = ()) -> str | tuple[str, ...]:
    """Normalize visibility and target_recipients into canonical action form.

    Returns:
        - "absent"
        - "global"
        - tuple of sorted agent IDs, e.g. ("agent_1",) or ("agent_1", "agent_2")
        - "targeted" (legacy fallback if recipients unknown)
    """
    if isinstance(visibility, (list, tuple)):
        clean = tuple(sorted(set(str(a) for a in visibility if a and a not in ("global", "absent"))))
        return clean if clean else "absent"
    if visibility == "global":
        return "global"
    if visibility == "absent" or not visibility:
        return "absent"
    if target_recipients:
        clean = tuple(sorted(set(str(a) for a in target_recipients if a and a not in ("global", "absent"))))
        return clean if clean else "absent"
    if visibility in ("targeted", "private"):
        return "targeted"
    if isinstance(visibility, str) and visibility.startswith("["):
        try:
            parsed = json.loads(visibility)
            if isinstance(parsed, list):
                clean = tuple(sorted(set(str(a) for a in parsed if a and a not in ("global", "absent"))))
                return clean if clean else "absent"
        except Exception:
            pass
    return "absent"


def serialize_action(action: Any) -> str:
    """Convert an action to its canonical string key for weight dicts and JSON."""
    if action == "absent" or action is None:
        return "absent"
    if action == "global":
        return "global"
    if action in ("targeted", "private"):
        return "targeted"
    if isinstance(action, (list, tuple)):
        clean = sorted(set(str(a) for a in action if a and a not in ("global", "absent")))
        return json.dumps(clean) if clean else "absent"
    if isinstance(action, str):
        if action.startswith("["):
            try:
                parsed = json.loads(action)
                if isinstance(parsed, list):
                    return serialize_action(parsed)
            except Exception:
                pass
        return json.dumps([action])
    return "absent"


def deserialize_action(action_str: str) -> str | tuple[str, ...]:
    """Parse a serialized action key back into canonical form."""
    if action_str in ("absent", "global", "targeted"):
        return action_str
    if action_str.startswith("["):
        try:
            parsed = json.loads(action_str)
            if isinstance(parsed, list):
                clean = tuple(sorted(set(str(a) for a in parsed if a and a not in ("global", "absent"))))
                return clean if clean else "absent"
        except Exception:
            pass
    return (action_str,)


def features(proposal: MemoryProposal, current_state: Sequence[MemoryItem]) -> Dict[str, float]:
    """Fixed feature extraction shared by training and inference."""
    title = proposal.title or ""
    val = proposal.raw_value or ""
    text = f"{title} {val}"
    return {
        "bias": 1.0,
        "shared_signal": 1.0 if _SHARED_SIGNALS.search(text) else 0.0,
        "private_signal": 1.0 if _PRIVATE_SIGNALS.search(text) else 0.0,
        "empty_value": 1.0 if not val.strip() else 0.0,
        "value_words": min(len(val.split()), 100) / 100.0,
        "has_supersedes": float(_find_supersedes(proposal, current_state) is not None),
        "is_action": 1.0 if _ACTION_SIGNALS.search(text) else 0.0,
        "turn_progress": min(proposal.step_index, 10) / 10.0,
    }


class LocalPolicyController:
    """Argmax-linear policy over candidate actions; weights persist as JSON."""

    FEATURE_KEYS = (
        "bias", "shared_signal", "private_signal", "empty_value",
        "value_words", "has_supersedes", "is_action", "turn_progress"
    )

    def __init__(
        self,
        weights: Dict[str, Dict[str, float]] | None = None,
        agent_role_map: Dict[str, str] | None = None,
    ) -> None:
        self.weights: Dict[str, Dict[str, float]] = {}
        raw_weights = dict(weights) if weights else {}
        for base_act in ("absent", "targeted", "global"):
            raw_act = "targeted" if (base_act == "targeted" and "targeted" not in raw_weights and "private" in raw_weights) else base_act
            self.weights[base_act] = {
                k: float(raw_weights.get(raw_act, {}).get(k, 0.0))
                for k in self.FEATURE_KEYS
            }
        for k, v in raw_weights.items():
            if k not in ("absent", "targeted", "global", "private"):
                norm_k = serialize_action(k)
                self.weights[norm_k] = {
                    fk: float(v.get(fk, 0.0)) for fk in self.FEATURE_KEYS
                }
        self.agent_role_map: Dict[str, str] = dict(agent_role_map or {})
        self.task_goal: str = ""

    def set_context(
        self,
        task_goal: str = "",
        agent_role_map: Any = None,
    ) -> None:
        if task_goal:
            self.task_goal = task_goal
        if agent_role_map is not None:
            self.agent_role_map = dict(agent_role_map)

    def score_base(self, action_key: str, feat: Dict[str, float]) -> float:
        w = self.weights.get(action_key, {})
        return sum(w.get(k, 0.0) * v for k, v in feat.items())

    def candidate_recipient_sets(self, proposal: MemoryProposal | None = None) -> List[tuple[str, ...]]:
        """Generate candidate recipient agent tuples from active agents and proposal."""
        known_agents = sorted(self.agent_role_map.keys())
        if not known_agents and proposal and proposal.agent_id:
            known_agents = [proposal.agent_id]

        recipients: List[tuple[str, ...]] = []
        if known_agents:
            recipients.extend((aid,) for aid in known_agents)
            if len(known_agents) <= 5:
                for r in range(2, len(known_agents) + 1):
                    recipients.extend(itertools.combinations(known_agents, r))
            else:
                recipients.extend(itertools.combinations(known_agents, 2))
                if proposal:
                    text = f"{proposal.title or ''} {proposal.raw_value or ''}".lower()
                    mentioned = tuple(sorted(set(aid for aid in known_agents if aid.lower() in text)))
                    if len(mentioned) >= 2:
                        recipients.append(mentioned)

        for k in self.weights:
            if k not in ("absent", "global", "targeted"):
                act = deserialize_action(k)
                if isinstance(act, tuple) and act not in recipients:
                    if not known_agents or all(aid in known_agents for aid in act):
                        recipients.append(act)

        if not recipients and proposal and proposal.agent_id:
            recipients.append((proposal.agent_id,))

        unique_recipients: List[tuple[str, ...]] = []
        seen = set()
        for r in recipients:
            if r not in seen:
                seen.add(r)
                unique_recipients.append(r)
        return unique_recipients

    def select_recipients(
        self,
        feat: Dict[str, float],
        proposal: MemoryProposal | None = None,
    ) -> tuple[str, ...]:
        """Select best recipient set when targeted visibility is chosen."""
        candidates = self.candidate_recipient_sets(proposal)
        if not candidates:
            return (proposal.agent_id,) if (proposal and proposal.agent_id) else ()

        text = f"{proposal.title or ''} {proposal.raw_value or ''}".lower() if proposal else ""

        def score_recipient_set(rec: tuple[str, ...]) -> float:
            ser = serialize_action(rec)
            if ser in self.weights:
                score = self.score_base(ser, feat)
            else:
                score = self.score_base("targeted", feat)

            bonus = 0.0
            for aid in rec:
                if aid.lower() in text:
                    bonus += 1.0
                role = self.agent_role_map.get(aid, "").lower()
                if role:
                    role_words = [w for w in re.findall(r"\w+", role) if len(w) > 2]
                    matches = sum(1 for w in role_words if w in text)
                    if matches > 0:
                        bonus += 0.5 * min(matches, 3)
                if feat.get("private_signal", 0.0) > 0.5 and proposal and aid == proposal.agent_id:
                    bonus += 0.3

            if len(rec) > 1 and bonus == 0.0:
                bonus -= 0.1 * len(rec)
            return score + bonus

        return max(candidates, key=score_recipient_set)

    def scores(self, feat: Dict[str, float], proposal: MemoryProposal | None = None) -> Dict[str, float]:
        res = {
            "absent": self.score_base("absent", feat),
            "targeted": self.score_base("targeted", feat),
            "global": self.score_base("global", feat),
        }
        for k in self.weights:
            if k not in res:
                res[k] = self.score_base(k, feat)
        return res

    def decide(
        self,
        proposal: MemoryProposal,
        current_state: Sequence[MemoryItem],
    ) -> MemoryTargetState:
        feat = features(proposal, current_state)
        score_dict = self.scores(feat, proposal)

        top_keys = ["absent", "targeted", "global"]
        for k in self.weights:
            if k not in top_keys:
                top_keys.append(k)

        if all(score_dict[k] == 0.0 for k in top_keys):
            best_key = "absent"
        else:
            best_key = max(top_keys, key=lambda k: score_dict[k])

        if best_key == "absent":
            return MemoryTargetState(
                exists=False,
                visibility="absent",
                target_recipients=(),
                supersedes=None,
            )

        supersedes = _find_supersedes(proposal, current_state)
        if best_key == "global":
            return MemoryTargetState(
                exists=True,
                visibility="global",
                target_recipients=(),
                supersedes=supersedes,
            )

        if best_key == "targeted":
            chosen_recipients = self.select_recipients(feat, proposal)
        else:
            parsed = deserialize_action(best_key)
            chosen_recipients = parsed if isinstance(parsed, tuple) else (proposal.agent_id,)

        return MemoryTargetState(
            exists=True,
            visibility="targeted",
            target_recipients=chosen_recipients,
            supersedes=supersedes,
        )

    def update(
        self,
        feat: Dict[str, float],
        label: Any,
        lr: float = 0.1,
        candidate_actions: Sequence[Any] | None = None,
    ) -> None:
        """One perceptron-style SFT step toward the labeled action."""
        target_action = canonical_action(label) if not isinstance(label, tuple) else label
        label_key = serialize_action(target_action)
        if label in ("targeted", "private"):
            label_key = "targeted"

        if label_key not in self.weights:
            self.weights[label_key] = {k: 0.0 for k in self.FEATURE_KEYS}

        if candidate_actions is not None:
            pool = [serialize_action(a) for a in candidate_actions]
        else:
            pool = list(self.weights.keys())
        if label_key not in pool:
            pool.append(label_key)

        for k in pool:
            if k not in self.weights:
                self.weights[k] = {fk: 0.0 for fk in self.FEATURE_KEYS}

        pred_key = max(pool, key=lambda k: sum(self.weights[k].get(fk, 0.0) * v for fk, v in feat.items()))

        if pred_key == label_key:
            return

        if pred_key in self.weights:
            for k, v in feat.items():
                self.weights[pred_key][k] = self.weights[pred_key].get(k, 0.0) - lr * v
        for k, v in feat.items():
            self.weights[label_key][k] = self.weights[label_key].get(k, 0.0) + lr * v

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.weights, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "LocalPolicyController":
        with open(path, encoding="utf-8") as fh:
            return cls(weights=json.load(fh))

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": self.weights}
