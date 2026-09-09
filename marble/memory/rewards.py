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
    outcome_gated: bool = False,
    pass_at_1: float | None = None,
    target_card_budget: int = 12,
    gamma_density: float = 0.0,
    task_success: bool | None = None,
    harmful_penalty: float = 0.2,
    advantage: float | None = None,
) -> Dict[str, float]:
    """G_i per proposal from one episode's trace events (Chapter 6 Credit Assignment).

    A = group_normalized_task_score or relative advantage (task_score - same_task_baseline).
    Credit rules:
    - ABSENT (or no memory_id): memory_credit = 0.0 (no free advantage for inaction)
    - Stored, unread: memory_credit = -lambda * cost - dp
    - Stored, author read: memory_credit = A - lambda * cost - dp
    - Stored, productive cross-agent read (reader != author, real memory, subsequent action, A > 0):
        memory_credit = A * (1 + alpha) - lambda * cost - dp
    - Stored, harmful cross-agent read (reader != author, A < 0):
        memory_credit = A - lambda * cost - dp - harmful_penalty
    """
    b = BETA if beta is None else beta
    lam = LAMBDA if lambda_ is None else lambda_
    proposals_data: List[Dict[str, Any]] = []
    global_cards_count = 0
    owners: Dict[str, str] = {}

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
        author = proposal.get("agent_id")
        if mid and author:
            owners[mid] = author
        tokens = float(ev.get("memory_cost_tokens", token_count(proposal.get("raw_value", ""))))
        proposals_data.append({
            "pid": pid,
            "mid": mid,
            "author": author,
            "owner": author,
            "tokens": tokens,
            "visibility": vis,
            "target_recipients": target.get("target_recipients", ()),
            "parse_status": ev.get("parse_status", "valid_json"),
        })

    for ev in events:
        if ev.get("event") in {"g_memory_operation", "memory_r1_operation", "collabmem_operation", "copper_operation"}:
            mid = ev.get("memory_id")
            aid = ev.get("agent_id") or (ev.get("proposal") or {}).get("agent_id")
            if mid and aid:
                owners[mid] = aid

    read_events = [ev for ev in events if ev.get("event") == "memory_read" and ev.get("memory_id")]
    readers: Dict[str, List[str]] = {}
    for ev in read_events:
        readers.setdefault(ev["memory_id"], []).append(ev["reader_id"])

    if advantage is not None:
        A = float(advantage)
    else:
        A = float(task_score) - float(same_task_baseline)

    # SimPO-style memory density penalty (only active when gamma_density > 0)
    density_penalty = 0.0
    if global_cards_count > target_card_budget and target_card_budget > 0 and gamma_density > 0:
        density_penalty = gamma_density * (global_cards_count - target_card_budget) / target_card_budget

    # Legacy outcome gating support if explicitly requested
    if task_success is not None:
        is_success = bool(task_success) if not isinstance(task_success, (int, float)) else (float(task_success) >= 1.0)
    elif pass_at_1 is not None:
        is_success = (pass_at_1 >= 1.0)
    else:
        is_success = (task_score >= 1.0)

    credits: Dict[str, float] = {}
    for p_info in proposals_data:
        pid = p_info["pid"]
        mid = p_info["mid"]
        vis = p_info["visibility"]

        # Rule 1: ABSENT or unsaved decisions get 0.0 credit (no free advantage)
        if vis == "absent" or not mid:
            credit_val = 0.0
        else:
            cost = p_info["tokens"] / _TOKEN_BUDGET
            was_read = mid in readers
            dp = density_penalty if vis == "global" else 0.0

            if not was_read:
                # Rule 2: Unread stored memory incurs storage cost and density penalty
                credit_val = -lam * cost - dp
            else:
                author = p_info.get("author") or p_info.get("owner") or owners.get(mid)
                mid_reads = [r for r in read_events if r.get("memory_id") == mid]
                non_author_reads = [r for r in mid_reads if r.get("reader_id") and r.get("reader_id") != author]

                if not non_author_reads:
                    # Rule 3: Read by author only
                    credit_val = A - lam * cost - dp
                else:
                    # Cross-agent read present: check if productive (subsequent action by reader)
                    is_effective = False
                    for r_ev in non_author_reads:
                        r_id = r_ev.get("reader_id")
                        try:
                            r_idx = events.index(r_ev)
                        except ValueError:
                            r_idx = -1
                        if r_idx >= 0 and r_idx < len(events) - 1:
                            for sub_ev in events[r_idx + 1:]:
                                sub_agent = (
                                    sub_ev.get("agent_id")
                                    or (sub_ev.get("proposal") or {}).get("agent_id")
                                    or (sub_ev.get("reader_id") if sub_ev.get("event") != "memory_read" else None)
                                )
                                if sub_agent == r_id:
                                    is_effective = True
                                    break
                        else:
                            # Terminal read in trace or synthetic unit test
                            is_effective = True
                        if is_effective:
                            break

                    # Rule 4 & 5: Effective bonus vs Harmful penalty
                    if outcome_gated and (not is_success or A <= 0):
                        credit_val = A - lam * cost - dp
                    elif is_effective and A > 0:
                        credit_val = A * (1.0 + b) - lam * cost - dp
                    elif A < 0:
                        credit_val = A - lam * cost - dp - harmful_penalty
                    else:
                        credit_val = A - lam * cost - dp

        if pid:
            credits[pid] = credit_val
        if mid:
            credits[mid] = credit_val

    return credits


