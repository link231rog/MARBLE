"""RL stage for the controller (spec §10): REINFORCE over stored traces.

Only the controller updates; workers/retriever/rewards stay frozen by design.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from marble.controllers.local_policy import VISIBILITIES, LocalPolicyController, features
from marble.memory.schema import MemoryProposal


def _as_proposal(d: Dict[str, Any]) -> MemoryProposal:
    return MemoryProposal(
        proposal_id=str(d.get("proposal_id", "replay")),
        task_id=str(d.get("task_id", "")),
        agent_id=str(d.get("agent_id", "")),
        source=str(d.get("source", "worker")),
        title=str(d.get("title", "")),
        raw_value=str(d.get("raw_value", "")),
        step_index=int(d.get("step_index", 0)),
    )


class RLTrainer:
    """Bandit-style update: proposals whose credit is positive are reinforced,
    negative-credit ones are pushed away from their chosen visibility.
    Includes SFT anchor regularization to prevent covariate drift (Chapter 3 §3.3)."""

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
        chosen = target.get("visibility")
        if chosen not in VISIBILITIES:
            return False
        feat = features(_as_proposal(proposal), [])
        if credit >= 0:
            self.controller.update(feat, chosen, lr=self.lr * credit)
            return True
        # negative advantage: move away from chosen toward the runner-up
        scores = {
            vis: sum(self.controller.weights[vis].get(k, 0.0) for k in feat)
            for vis in VISIBILITIES
        }
        others = [v for v in VISIBILITIES if v != chosen]
        runner_up = max(others, key=lambda v: scores[v])
        self.controller.update(feat, runner_up, lr=self.lr * (-credit))
        return True

    def apply_anchor(self) -> None:
        """Chapter 3 §3.3: Anchor policy update to SFT prior to avoid covariate drift."""
        if not self.initial_weights or self.anchor_coeff <= 0:
            return
        for vis in VISIBILITIES:
            init_map = self.initial_weights.get(vis, {})
            for k, init_val in init_map.items():
                curr_val = self.controller.weights[vis].get(k, 0.0)
                self.controller.weights[vis][k] = curr_val - self.anchor_coeff * (curr_val - init_val)

    def update_trace(self, events: List[Dict[str, Any]], credits: Dict[str, float]) -> int:
        steps = 0
        for ev in events:
            if ev.get("event") != "memory_decision":
                continue
            mid = ev.get("memory_id")
            if mid and mid in credits and self.update_event(ev, credits[mid]):
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
    gamma_density: float = 0.02,
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
        vis: dict(controller.weights.get(vis, {}))
        for vis in VISIBILITIES
    }

    trainer = RLTrainer(
        controller,
        lr=lr,
        initial_weights=init_weights_copy,
        anchor_coeff=anchor_coeff,
    )
    from marble.memory.rewards import proposal_rewards

    # Auto-resolve task scores and group-relative baseline (Chapter 3 §3.2)
    if task_scores is None or len(task_scores) != len(trace_paths):
        collected = []
        for path in trace_paths:
            summary_p = Path(path).with_name("summary.json")
            score = r_episode
            if summary_p.is_file():
                try:
                    with summary_p.open(encoding="utf-8") as sfh:
                        score = float(json.load(sfh).get("task_score", r_episode))
                except Exception:
                    pass
            collected.append(score)
        task_scores = collected

    if same_task_baseline == 0.0 and task_scores and len(task_scores) > 1:
        same_task_baseline = sum(task_scores) / len(task_scores)

    episodes = 0
    steps = 0
    for i, path in enumerate(trace_paths):
        with open(path, encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        decisions = [e for e in events if e.get("event") == "memory_decision"]
        if not decisions:
            continue
        episodes += 1
        ts = task_scores[i] if task_scores and i < len(task_scores) else r_episode

        credits = proposal_rewards(
            events,
            task_score=ts,
            same_task_baseline=same_task_baseline,
            outcome_gated=outcome_gated,
            target_card_budget=target_card_budget,
            gamma_density=gamma_density,
        )
        for _ in range(epochs):
            steps += trainer.update_trace(events, credits)
    controller.save(out_path)
    stats = {"episodes": episodes, "updates": steps, "out": out_path}
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

