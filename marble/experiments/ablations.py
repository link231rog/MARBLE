"""One-factor-at-a-time ablations (spec §14).

An ablation is a ``factor:option`` string. Transforms are pure so the CLI can
apply them before building controllers/configs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from marble.memory.schema import MemoryProposal, MemoryTargetState

FACTORS = ("policy", "schema_field", "retrieval", "state_update", "reward", "training", "input")

SCHEMA_FIELDS = ("title", "value", "source", "agent_id", "task_id", "step_index")

# Input-block ablation -> JsonController drop_fields (schema-and-reward.md §Input ablations)
INPUT_ABLATIONS = {
    "no_agent_tags": ("agent_capabilities",),
    "no_topic_tags": ("topics",),
    "no_memory_summary": ("memory_summary",),
    "no_active_memory_index": ("active_memory_index",),
}


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
    if factor == "input":
        if option not in INPUT_ABLATIONS:
            raise ValueError(f"input option must be one of {sorted(INPUT_ABLATIONS)}")
        return {"drop_fields": INPUT_ABLATIONS[option]}
    return {}


def wrap_controller(controller, factor: str, option: str):
    if factor == "state_update" and option == "off":
        return NoSupersede(controller)
    return controller


def reward_override(factor: str, option: str) -> Dict[str, float]:
    """Passed to proposal_rewards at evaluation/training time."""
    if factor == "reward" and option == "beta0":
        return {"beta": 0.0}
    if factor == "reward" and option == "lambda0":
        return {"lambda_": 0.0}
    return {}


# ----------------------------------------------------------------------- CLI
def _load_config(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        return json.loads(text)


def run_ablation(base_config: str, factor: str, option: str, out: str) -> Dict[str, Any]:
    cfg = _load_config(base_config)
    cfg = apply_to_task_config(cfg, factor, option)
    text = _dump_config(cfg)
    Path(out).write_text(text, encoding="utf-8")
    return {"out": out, "ablation": f"{factor}:{option}"}


def _dump_config(cfg: Dict[str, Any]) -> str:
    try:
        import yaml

        return yaml.safe_dump(cfg, allow_unicode=True)
    except ImportError:
        return json.dumps(cfg, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Apply a one-factor ablation to a task config")
    ap.add_argument("--base-config", required=True, help="path to a task config yaml/json")
    ap.add_argument("--factor", required=True, choices=FACTORS)
    ap.add_argument("--option", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", action="store_true",
                    help="also run the benchmark via run_benchmark (reuses its pipeline)")
    ap.add_argument("--benchmark", default="coding", help="benchmark to run when --run")
    ap.add_argument("--baseline", default="learned_controller", help="baseline when --run")
    ap.add_argument("--out-dir", default="runs/ablations", help="run dir when --run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    res = run_ablation(args.base_config, args.factor, args.option, args.out)
    print(json.dumps(res))
    if args.run:
        from marble.experiments import run_benchmark

        argv = [
            "--benchmark", args.benchmark,
            "--baseline", args.baseline,
            "--ablation", f"{args.factor}:{args.option}",
            "--out", args.out_dir,
        ]
        if args.dry_run:
            argv.append("--dry-run")
        run_benchmark.main(argv)
