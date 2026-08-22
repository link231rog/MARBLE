"""Explainable episode reward and per-proposal delayed credit (spec §Reward)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

REUSE_WEIGHT_NON_OWNER = 1.25
REUSE_WEIGHT_OWNER = 1.00

W_COORDINATION = 0.25
W_READ_COST = 0.10
W_GLOBAL_STORAGE = 0.05
W_COMM_COST = 0.10


def token_count(text: str) -> int:
    # ponytail: whitespace-word proxy for tokens; swap in a tokenizer when budgets matter
    return len(text.split())


@dataclass
class EpisodeStats:
    task_score: float = 0.0
    coordination_score: float = 0.0
    read_tokens: int = 0
    active_global_tokens: int = 0
    communication_tokens: int = 0
    token_budget: int = 4096


def episode_reward(stats: EpisodeStats) -> float:
    budget = max(stats.token_budget, 1)
    return (
        stats.task_score
        + W_COORDINATION * stats.coordination_score
        - W_READ_COST * stats.read_tokens / budget
        - W_GLOBAL_STORAGE * stats.active_global_tokens / budget
        - W_COMM_COST * stats.communication_tokens / budget
    )


def proposal_rewards(
    events: List[Dict[str, Any]],
    r_episode: float,
    running_baseline: float = 0.0,
    reuse_weight_non_owner: float | None = None,
    reuse_weight_owner: float | None = None,
) -> Dict[str, float]:
    """G_t per stored proposal from one episode's trace events.

    G_t = -storage_cost + I[read] * reuse_weight * (r_episode - running_baseline)
    Weight overrides exist for the reward ablation (spec §14).
    """
    w_non_owner = REUSE_WEIGHT_NON_OWNER if reuse_weight_non_owner is None else reuse_weight_non_owner
    w_owner = REUSE_WEIGHT_OWNER if reuse_weight_owner is None else reuse_weight_owner
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

    advantage = r_episode - running_baseline
    credits: Dict[str, float] = {}
    for mid, info in stored.items():
        storage_cost = info["tokens"] / 4096  # same budget scale as EpisodeStats default
        was_read = mid in readers
        if not was_read:
            credits[mid] = -storage_cost
            continue
        non_owner_reads = [r for r in readers[mid] if r != info["owner"]]
        weight = w_non_owner if non_owner_reads else w_owner
        credits[mid] = -storage_cost + weight * advantage
    return credits
