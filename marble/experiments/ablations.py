"""One-factor-at-a-time ablations (spec §14).

An ablation is a ``factor:option`` string. Transforms are pure so the CLI can
apply them before building controllers/configs.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

from marble.memory.schema import MemoryProposal, MemoryTargetState

FACTORS = ("policy", "schema_field", "retrieval", "state_update", "reward", "training")

SCHEMA_FIELDS = ("title", "value", "source", "agent_id", "task_id", "step_index")


def parse_ablation(spec: str) -> tuple:
    """'state_update:off' -> ('state_update', 'off')."""
    factor, _, option = spec.partition(":")
    if factor not in FACTORS or not option:
        raise ValueError(f"ablation must be factor:option with factor in {FACTORS}")
    return factor, option


class NoSupersede:
    """Wraps a controller; forces supersedes=None (state-update ablation)."""

    def __init__(self, inner):
        self.inner = inner

    def decide(self, proposal: MemoryProposal, current_state) -> MemoryTargetState:
        target = self.inner.decide(proposal, current_state)
        return MemoryTargetState(
            exists=target.exists,
            visibility=target.visibility,
            owner_id=target.owner_id,
            supersedes=None,
        )


def apply_to_task_config(cfg: Dict[str, Any], factor: str, option: str) -> Dict[str, Any]:
    """Config-level knobs only; controller-level ones live in controller_kwargs."""
    cfg = {**cfg, "memory": {**cfg.get("memory", {})}}
    if factor == "policy":
        cfg["memory"]["controller"] = option
    elif factor == "retrieval" and option == "none":
        cfg["memory"]["max_cards"] = 0
    elif factor == "state_update":
        cfg["memory"]["supersedes_enabled"] = option != "off"
    cfg["memory"]["ablation"] = f"{factor}:{option}"
    return cfg


def controller_kwargs(factor: str, option: str) -> Dict[str, Any]:
    """Extra kwargs for JsonController / wrapper selection."""
    if factor == "schema_field":
        if option not in SCHEMA_FIELDS:
            raise ValueError(f"schema_field option must be one of {SCHEMA_FIELDS}")
        return {"drop_fields": (option,)}
    return {}


def wrap_controller(controller, factor: str, option: str):
    if factor == "state_update" and option == "off":
        return NoSupersede(controller)
    return controller


def reward_override(factor: str, option: str) -> Dict[str, float]:
    """Passed to proposal_rewards at evaluation/training time."""
    if factor == "reward" and option == "no_reuse":
        return {"reuse_weight_non_owner": 0.0, "reuse_weight_owner": 0.0}
    return {}
