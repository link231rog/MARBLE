"""Linear-policy controller trainable via offline SFT (spec Stage 3)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

from marble.memory.schema import MemoryItem, MemoryProposal, MemoryTargetState

VISIBILITIES = ("absent", "private", "global")

_SHARED_SIGNALS = re.compile(
    r"\b(share|shared|all|team|decision|result|consensus|plan|finding)\b", re.IGNORECASE
)


def features(proposal: MemoryProposal, current_state: Sequence[MemoryItem]) -> Dict[str, float]:
    """Fixed feature extraction shared by training and inference."""
    return {
        "bias": 1.0,
        "shared_signal": 1.0 if _SHARED_SIGNALS.search(proposal.title) else 0.0,
        "empty_value": 1.0 if not proposal.raw_value.strip() else 0.0,
        "value_words": min(len(proposal.raw_value.split()), 100) / 100.0,
        "has_supersedes": float(
            any(item.active and item.title.casefold() == proposal.title.casefold()
                for item in current_state)
        ),
    }


class LocalPolicyController:
    """Argmax-linear policy over features; weights persist as JSON."""

    def __init__(self, weights: Dict[str, Dict[str, float]] | None = None) -> None:
        self.weights = weights or {vis: {k: 0.0 for k in (
            "bias", "shared_signal", "empty_value", "value_words", "has_supersedes")}
            for vis in VISIBILITIES}

    def scores(self, feat: Dict[str, float]) -> Dict[str, float]:
        return {vis: sum(w.get(k, 0.0) * v for k, v in feat.items())
                for vis, w in self.weights.items()}

    def decide(self, proposal: MemoryProposal,
               current_state: Sequence[MemoryItem]) -> MemoryTargetState:
        feat = features(proposal, current_state)
        vis = max(VISIBILITIES, key=lambda v: self.scores(feat)[v])
        exists = vis != "absent"
        return MemoryTargetState(
            exists=exists,
            visibility=vis if exists else "absent",
            owner_id=proposal.agent_id if vis == "private" else None,
        )

    def update(self, feat: Dict[str, float], label: str, lr: float = 0.1) -> None:
        """One perceptron-style SFT step toward the labeled visibility."""
        pred = max(VISIBILITIES, key=lambda v: self.scores(feat)[v])
        if pred == label:
            return
        for vis in VISIBILITIES:
            sign = -1.0 if vis == pred else (1.0 if vis == label else 0.0)
            for k, v in feat.items():
                self.weights[vis][k] += lr * sign * v

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.weights, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "LocalPolicyController":
        with open(path, encoding="utf-8") as fh:
            return cls(weights=json.load(fh))

    def to_dict(self) -> Dict[str, Any]:
        return {"weights": self.weights}
