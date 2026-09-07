"""RL stage for the controller (spec §10): REINFORCE over stored traces.

Only the controller updates; workers/retriever/rewards stay frozen by design.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from marble.controllers.local_policy import (
    VISIBILITIES,
    LocalPolicyController,
    canonical_action,
    features,
    serialize_action,
)
from marble.memory.schema import MemoryItem, MemoryProposal


def _as_proposal(p: Dict[str, Any]) -> MemoryProposal:
    return MemoryProposal(
        proposal_id=str(p.get("proposal_id") or "p"),
        task_id=str(p.get("task_id") or "t"),
        agent_id=str(p.get("agent_id") or "a"),
        source=str(p.get("source") or "worker"),
        title=str(p.get("title") or ""),
        raw_value=str(p.get("raw_value") or ""),
        step_index=int(p.get("step_index") or 0),
    )


def _as_memory_item(d: Dict[str, Any]) -> MemoryItem | None:
    try:
        vis = d.get("visibility", "global")
        if vis not in ("global", "targeted"):
            vis = "targeted" if isinstance(vis, list) else "global"
        return MemoryItem(
            memory_id=str(d.get("memory_id") or "m"),
            proposal_id=str(d.get("proposal_id") or "p"),
            task_id=str(d.get("task_id") or "t"),
            title=str(d.get("title") or ""),
            raw_value=str(d.get("raw_value") or ""),
            visibility=vis,
            source_agent=str(d.get("source_agent") or "a"),
            source=str(d.get("source") or "worker"),
            step_index=int(d.get("step_index") or 0),
            active=bool(d.get("active", True)),
            supersedes=d.get("supersedes"),
            created_at=int(d.get("created_at") or 0),
            target_recipients=tuple(d.get("target_recipients") or ()),
        )
    except Exception:
        return None


class RLTrainer:
    """Online policy iteration for LocalPolicyController using REINFORCE-style updates."""

    def __init__(
        self,
        controller: LocalPolicyController,
        lr: float = 0.05,
        initial_weights: Dict[str, Dict[str, float]] | None = None,
        anchor_coeff: float = 0.0,
    ):
        self.controller = controller
        self.lr = lr
        self.initial_weights = initial_weights
        self.anchor_coeff = anchor_coeff

    def update_event(self, event: Dict[str, Any], credit: float) -> bool:
        proposal = event.get("proposal")
        target = event.get("target")
        if not proposal or not target:
            return False
        vis = target.get("visibility")
        recipients = target.get("target_recipients") or ()
        if vis not in ("absent", "global", "targeted", "private") and not isinstance(vis, (list, tuple)) and not recipients:
            return False
        act = canonical_action(vis, recipients)
        if not target.get("exists", True) and act != "absent":
            act = "absent"
        chosen = serialize_action(act)

        raw_active = event.get("active_memory_index") or []
        active_items = [it for it in (_as_memory_item(x) for x in raw_active) if it is not None]
        prop_obj = _as_proposal(proposal)
        feat = features(prop_obj, active_items)
        if credit >= 0:
            self.controller.update(feat, chosen, lr=self.lr * credit)
            return True
        # negative advantage: move away from chosen toward the runner-up
        candidates = sorted(set(self.controller.weights.keys()) | {"absent", "global", "targeted"})
        scores = self.controller.scores(feat, prop_obj)
        others = [c for c in candidates if c != chosen]
        if not others:
            return False
        runner_up = max(others, key=lambda c: scores.get(c, 0.0))
        self.controller.update(feat, runner_up, lr=self.lr * (-credit), candidate_actions=candidates)
        return True

    def apply_anchor(self) -> None:
        """Chapter 3 §3.3: Anchor policy update to SFT prior to avoid covariate drift."""
        if not self.initial_weights or self.anchor_coeff <= 0:
            return
        for act in self.initial_weights:
            init_map = self.initial_weights.get(act, {})
            curr_map = self.controller.weights.setdefault(act, {k: 0.0 for k in LocalPolicyController.FEATURE_KEYS})
            for k, init_val in init_map.items():
                curr_val = curr_map.get(k, 0.0)
                curr_map[k] = curr_val - self.anchor_coeff * (curr_val - init_val)

    def update_trace(self, events: List[Dict[str, Any]], credits: Dict[str, float]) -> int:
        steps = 0
        for ev in events:
            if ev.get("event") != "memory_decision":
                continue
            if ev.get("parse_status") in ("format_error", "schema_error"):
                continue
            pid = ev.get("proposal", {}).get("proposal_id")
            mid = ev.get("memory_id")
            credit = credits.get(pid) if pid else None
            if credit is None and mid:
                credit = credits.get(mid)
            if credit is not None and self.update_event(ev, credit):
                steps += 1
        if steps > 0:
            self.apply_anchor()
        return steps


def train_rl(
    trace_paths: List[str],
    out_path: str,
    init_checkpoint: str | None = None,
    epochs: int = 5,
    lr: float = 0.05,
    r_episode: float = 1.0,
    task_scores: List[float] | None = None,
    same_task_baseline: float = 0.0,
    outcome_gated: bool = True,
    target_card_budget: int = 12,
    gamma_density: float = 0.0,
    anchor_coeff: float = 0.005,
) -> Dict[str, Any]:
    # Resolve warm-start checkpoint (Chapter 3 §2.1 & §3.1)
    controller = None
    if init_checkpoint and init_checkpoint.lower() not in ("scratch", "none", "null"):
        controller = LocalPolicyController.load(init_checkpoint)
    elif init_checkpoint is None:
        sft_cand = Path("runs/ours_sft_policy.json")
        if sft_cand.is_file():
            controller = LocalPolicyController.load(str(sft_cand))
    if controller is None:
        controller = LocalPolicyController()

    init_weights_copy = {
        act: dict(weights)
        for act, weights in controller.weights.items()
    }

    trainer = RLTrainer(
        controller,
        lr=lr,
        initial_weights=init_weights_copy,
        anchor_coeff=anchor_coeff,
    )
    from marble.memory.rewards import proposal_rewards

    # Auto-resolve task scores and group-relative baseline per (benchmark, task_id)
    episodes: List[Dict[str, Any]] = []
    scores_by_task: Dict[Tuple[str, str], List[float]] = {}
    for i, path in enumerate(trace_paths):
        summary_p = Path(path).with_name("summary.json")
        score = task_scores[i] if task_scores and i < len(task_scores) else r_episode
        benchmark = ""
        task_id = ""
        if summary_p.is_file():
            try:
                with summary_p.open(encoding="utf-8") as sfh:
                    s_data = json.load(sfh)
                    if task_scores is None or i >= len(task_scores):
                        score = float(s_data.get("task_score", r_episode))
                    benchmark = str(s_data.get("benchmark", ""))
                    task_id = str(s_data.get("task_id", ""))
            except Exception:
                pass
        with open(path, encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        if not task_id:
            for ev in events:
                if ev.get("event") == "memory_decision":
                    task_id = str(ev.get("proposal", {}).get("task_id", ""))
                    if task_id:
                        break
        task_key = (benchmark, task_id or f"task_{i}")
        scores_by_task.setdefault(task_key, []).append(score)
        episodes.append({"events": events, "score": score, "task_key": task_key})

    task_baselines: Dict[Tuple[str, str], float] = {}
    for task_key, scores in scores_by_task.items():
        if same_task_baseline > 0.0:
            task_baselines[task_key] = same_task_baseline
        elif len(scores) >= 2:
            task_baselines[task_key] = sum(scores) / len(scores)
        else:
            task_baselines[task_key] = 0.0

    episodes_count = 0
    steps = 0
    for ep in episodes:
        events = ep["events"]
        decisions = [e for e in events if e.get("event") == "memory_decision"]
        if not decisions:
            continue
        episodes_count += 1
        ts = ep["score"]
        base = task_baselines[ep["task_key"]]

        credits = proposal_rewards(
            events,
            task_score=ts,
            same_task_baseline=base,
            outcome_gated=outcome_gated,
            target_card_budget=target_card_budget,
            gamma_density=gamma_density,
        )
        for _ in range(epochs):
            steps += trainer.update_trace(events, credits)
    controller.save(out_path)
    stats = {"episodes": episodes_count, "updates": steps, "out": out_path}
    print(json.dumps(stats))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RL fine-tune of the local policy controller")
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--init", default=None, help="SFT checkpoint to start from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--r-episode", type=float, default=1.0)
    ap.add_argument("--same-task-baseline", type=float, default=0.0)
    ap.add_argument("--outcome-gated", action="store_true", default=True)
    ap.add_argument("--target-card-budget", type=int, default=12)
    ap.add_argument("--anchor-coeff", type=float, default=0.005)
    args = ap.parse_args()
    train_rl(
        args.traces,
        args.out,
        init_checkpoint=args.init,
        epochs=args.epochs,
        lr=args.lr,
        r_episode=args.r_episode,
        same_task_baseline=args.same_task_baseline,
        outcome_gated=args.outcome_gated,
        target_card_budget=args.target_card_budget,
        anchor_coeff=args.anchor_coeff,
    )

