"""RL stage for the controller (spec §10): REINFORCE over stored traces.

Only the controller updates; workers/retriever/rewards stay frozen by design.
"""
from __future__ import annotations

import argparse
import json
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
    negative-credit ones are pushed away from their chosen visibility."""

    def __init__(self, controller: LocalPolicyController, lr: float = 0.05):
        self.controller = controller
        self.lr = lr

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

    def update_trace(self, events: List[Dict[str, Any]], credits: Dict[str, float]) -> int:
        steps = 0
        for ev in events:
            if ev.get("event") != "memory_decision":
                continue
            mid = ev.get("memory_id")
            if mid and mid in credits and self.update_event(ev, credits[mid]):
                steps += 1
        return steps


def train_rl(
    trace_paths: List[str],
    out_path: str,
    init_checkpoint: str | None = None,
    epochs: int = 5,
    lr: float = 0.05,
    r_episode: float = 1.0,
) -> Dict[str, Any]:
    controller = (
        LocalPolicyController.load(init_checkpoint)
        if init_checkpoint
        else LocalPolicyController()
    )
    trainer = RLTrainer(controller, lr=lr)
    from marble.memory.rewards import proposal_rewards

    episodes = 0
    steps = 0
    for path in trace_paths:
        with open(path, encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
        decisions = [e for e in events if e.get("event") == "memory_decision"]
        if not decisions:
            continue
        episodes += 1
        # ponytail: running-baseline omitted — single-method offline replay only
        credits = proposal_rewards(events, task_score=r_episode)
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
    args = ap.parse_args()
    train_rl(args.traces, args.out, init_checkpoint=args.init, epochs=args.epochs, lr=args.lr)
