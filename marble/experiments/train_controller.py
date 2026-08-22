"""Offline SFT warm-up: fit LocalPolicyController weights from decision traces."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from marble.controllers import LocalPolicyController, features
from marble.controllers.local_policy import VISIBILITIES
from marble.memory.schema import MemoryProposal


def load_samples(trace_paths: List[str]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for path in trace_paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                ev = json.loads(line)
                if ev.get("event") != "memory_decision" or not ev.get("memory_id"):
                    continue
                p = ev["proposal"]
                proposal = MemoryProposal(
                    proposal_id=p["proposal_id"], task_id=p["task_id"],
                    agent_id=p["agent_id"], source=p.get("source", "worker"),
                    title=p["title"], raw_value=p["raw_value"],
                    step_index=p.get("step_index", 0),
                )
                # ponytail: supersedes context not replayed here; add when traces store state snapshots
                samples.append({
                    "feat": features(proposal, []),
                    "label": ev["target"]["visibility"],
                })
    return samples


def accuracy(policy: LocalPolicyController, samples: List[Dict[str, Any]]) -> float:
    if not samples:
        return 0.0
    hits = sum(
        max(VISIBILITIES,
            key=lambda v: policy.scores(s["feat"])[v]) == s["label"]
        for s in samples
    )
    return hits / len(samples)


def train(trace_paths: List[str], out_path: str, epochs: int = 20,
          lr: float = 0.1) -> Dict[str, Any]:
    samples = load_samples(trace_paths)
    policy = LocalPolicyController()
    before = accuracy(policy, samples)
    for _ in range(epochs):
        for s in samples:
            policy.update(s["feat"], s["label"], lr=lr)
    after = accuracy(policy, samples)
    policy.save(out_path)
    stats = {"samples": len(samples), "accuracy_before": before,
             "accuracy_after": after, "out": out_path}
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT the local policy controller.")
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--out", default="runs/local_policy.json")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(args.traces, args.out, epochs=args.epochs)
