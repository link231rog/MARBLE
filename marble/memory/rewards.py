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

    @property
    def memory_cost(self) -> float:
        budget = max(self.token_budget, 1)
        return (
            self.read_tokens + self.active_global_tokens + self.communication_tokens
        ) / budget


def episode_reward(stats: EpisodeStats) -> float:
    return stats.task_score - LAMBDA * stats.memory_cost


def measured_memory_cost(events: List[Dict[str, Any]], token_budget: int = _TOKEN_BUDGET) -> float:
    """Return variable memory-context tokens from one episode trace."""
    total = 0
    for event in events:
        if event.get("event") != "memory_decision":
            continue
        total += int(event.get("memory_card_tokens", 0) or 0)
        total += int(event.get("injected_memory_tokens", 0) or 0)
    return total / max(token_budget, 1)


def proposal_rewards(
    events: List[Dict[str, Any]],
    task_score: float = 0.0,
    same_task_baseline: float = 0.0,
    beta: float | None = None,
    lambda_: float | None = None,
    outcome_gated: bool = True,
    pass_at_1: float | None = None,
    target_card_budget: int = 12,
    gamma_density: float = 0.0,
    task_success: bool | None = None,
) -> Dict[str, float]:
    """G_i per proposal from one episode's trace events (doc §Reward, Chapter 3).

    A_e = task_score - same_task_baseline (same-task relative advantage).
    With outcome_gated=True (2025 RLVR standard), non-owner collaboration bonus (1+beta)
    is only awarded if the task succeeds and advantage > 0.
    For absent decisions, memory cost is 0.0 and credit equals the episode advantage.
    Credits are keyed by both proposal_id and memory_id for seamless joins.
    """
    b = BETA if beta is None else beta
    lam = LAMBDA if lambda_ is None else lambda_
    proposals_data: List[Dict[str, Any]] = []
    global_cards_count = 0
    for ev in events:
        if ev.get("event") != "memory_decision":
            continue
        mid = ev.get("memory_id")
        target = ev.get("target") or {}
        vis = target.get("visibility")
        if vis == "global" and mid:
            global_cards_count += 1
        proposal = ev.get("proposal") or {}
        pid = proposal.get("proposal_id")
        tokens = float(ev.get("memory_cost_tokens", token_count(proposal.get("raw_value", ""))))
        proposals_data.append({
            "pid": pid,
            "mid": mid,
            "owner": proposal.get("agent_id"),
            "tokens": tokens,
            "visibility": vis,
            "parse_status": ev.get("parse_status", "valid_json"),
        })

    readers: Dict[str, List[str]] = {}
    for ev in events:
        if ev.get("event") == "memory_read":
            readers.setdefault(ev["memory_id"], []).append(ev["reader_id"])

    advantage = task_score - same_task_baseline

    # SimPO-style memory density penalty (only active when gamma_density > 0)
    density_penalty = 0.0
    if global_cards_count > target_card_budget and target_card_budget > 0 and gamma_density > 0:
        density_penalty = gamma_density * (global_cards_count - target_card_budget) / target_card_budget

    # RLVR Outcome Gating: only successful episodes get collaboration multiplier (Chapter 3 §2.2)
    if task_success is not None:
        is_success = bool(task_success)
    elif pass_at_1 is not None:
        is_success = (pass_at_1 >= 1.0)
    else:
        is_success = (task_score >= 1.0)

    credits: Dict[str, float] = {}
    for p_info in proposals_data:
        pid = p_info["pid"]
        mid = p_info["mid"]
        vis = p_info["visibility"]
        if vis == "absent" or not mid:
            credit_val = advantage
        else:
            cost = p_info["tokens"] / _TOKEN_BUDGET
            was_read = mid in readers
            if not was_read:
                credit_val = -lam * cost - density_penalty
            else:
                non_owner = any(r != p_info["owner"] for r in readers[mid])
                if outcome_gated and (not is_success or advantage <= 0):
                    mult = 1.0
                else:
                    mult = 1.0 + b * (1 if non_owner else 0)
                credit_val = advantage * mult - lam * cost - density_penalty

        if pid:
            credits[pid] = credit_val
        if mid:
            credits[mid] = credit_val

    return credits

