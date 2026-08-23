"""Episode reward and per-memory credit (schema-and-reward.md).

R_episode = task_score - lambda * memory_cost
G_i = I(read) * A_e * (1 + beta * I(read by non-owner)) - lambda * memory_i_cost

This document is the source of truth for reward; earlier multi-term drafts are
not. lambda/beta are starting values, not optimality claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

LAMBDA = 0.05  # memory-cost weight
BETA = 0.25    # cross-agent reuse multiplier
_TOKEN_BUDGET = 4096


def token_count(text: str) -> int:
    # ponytail: whitespace-word proxy for tokens; swap in a tokenizer when budgets matter
    return len(text.split())


@dataclass
class EpisodeStats:
    task_score: float = 0.0
    read_tokens: int = 0
    active_global_tokens: int = 0
    communication_tokens: int = 0
    token_budget: int = _TOKEN_BUDGET


def episode_reward(stats: EpisodeStats) -> float:
    budget = max(stats.token_budget, 1)
    memory_cost = (
        stats.read_tokens + stats.active_global_tokens + stats.communication_tokens
    ) / budget
    return stats.task_score - LAMBDA * memory_cost


def proposal_rewards(
    events: List[Dict[str, Any]],
    task_score: float = 0.0,
    same_task_baseline: float = 0.0,
    beta: float | None = None,
    lambda_: float | None = None,
) -> Dict[str, float]:
    """G_i per stored proposal from one episode's trace events (doc §Reward).

    A_e = task_score - same_task_baseline (same-task relative advantage).
    With no comparable rollout, pass same_task_baseline=0 (no advantage signal).
    """
    b = BETA if beta is None else beta
    lam = LAMBDA if lambda_ is None else lambda_
    stored: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        if ev.get("event") != "memory_decision":
            continue
        mid = ev.get("memory_id")
        if not mid:
            continue
        proposal = ev["proposal"]
        stored[mid] = {
            "owner": proposal["agent_id"],
            "tokens": token_count(proposal.get("raw_value", "")),
        }
    readers: Dict[str, List[str]] = {}
    for ev in events:
        if ev.get("event") == "memory_read":
            readers.setdefault(ev["memory_id"], []).append(ev["reader_id"])

    advantage = task_score - same_task_baseline
    credits: Dict[str, float] = {}
    for mid, info in stored.items():
        cost = info["tokens"] / _TOKEN_BUDGET
        was_read = mid in readers
        if not was_read:
            credits[mid] = -lam * cost
            continue
        non_owner = any(r != info["owner"] for r in readers[mid])
        mult = 1 + b * (1 if non_owner else 0)
        credits[mid] = advantage * mult - lam * cost
    return credits
