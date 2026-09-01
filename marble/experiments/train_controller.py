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
    parser = argparse.ArgumentParser(description="Train a controller from decision traces.")
    parser.add_argument("--mode", choices=("sft", "rl", "qwen_sft", "qwen_rl", "sft-qwen", "rl-qwen"),
                        default="sft",
                        help="sft/rl: linear LocalPolicy; qwen_sft: LoRA SFT; "
                             "qwen_rl: trace-replay completion-level REINFORCE")
    parser.add_argument("--traces", nargs="+", required=True)
    parser.add_argument("--out", default="runs/local_policy.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init", default=None, help="SFT checkpoint to start RL from")
    parser.add_argument("--r-episode", type=float, default=1.0,
                        help="fallback episode reward when --rewards is omitted")
    parser.add_argument("--rewards", nargs="+", default=None,
                        help="summary.json paths (parallel to --traces) providing real "
                             "task_score for the RL credit loop")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="base model for Qwen LoRA training")
    args = parser.parse_args()
    if args.mode in ("qwen_sft", "qwen_rl", "sft-qwen", "rl-qwen"):
        from marble.controllers import qwen_lora

        if args.mode in ("qwen_sft", "sft-qwen"):
            pairs = qwen_lora.export_sft_pairs(args.traces)
            qwen_lora.train_qwen_sft(pairs, args.out, args.base_model, epochs=args.epochs)
        else:
            if not args.rewards:
                parser.error("--rewards is required for qwen_rl")
            rewards = [
                float(json.load(open(reward_path, encoding="utf-8")).get("task_score", 0.0))
                for reward_path in args.rewards
            ]
            qwen_lora.train_qwen_rl(
                args.traces, args.out, args.base_model, rewards=rewards,
                epochs=args.epochs, init_checkpoint=args.init,
            )
    elif args.mode == "rl":
        from marble.controllers.rl_controller import train_rl

        task_scores = None
        baseline = 0.0
        if args.rewards:
            task_scores = []
            for rp in args.rewards:
                with open(rp, encoding="utf-8") as fh:
                    task_scores.append(float(json.load(fh).get("task_score", 0.0)))
            baseline = sum(task_scores) / len(task_scores) if task_scores else 0.0
        train_rl(args.traces, args.out, init_checkpoint=args.init,
                 epochs=args.epochs, lr=0.05, r_episode=args.r_episode,
                 task_scores=task_scores, same_task_baseline=baseline)
    else:
        train(args.traces, args.out, epochs=args.epochs)
