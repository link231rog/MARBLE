"""Offline SFT warm-up: fit LocalPolicyController weights from decision traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from marble.controllers.local_policy import (
    LocalPolicyController,
    canonical_action,
    features,
    serialize_action,
)
from marble.experiments.baselines import canonical_baseline
from marble.experiments.task_manifest import read_manifest
from marble.memory.schema import MemoryProposal


def _in_split(task_id: Any, split: str) -> bool:
    if split == "all":
        return True
    bucket = int(hashlib.sha256(str(task_id).encode()).hexdigest(), 16) % 10
    return (bucket < 8) == (split in ("train", "train_hard"))


def _canonical_or_raw(name: Any) -> str:
    try:
        return canonical_baseline(str(name))
    except ValueError:
        return str(name)


def _manifest_keys(manifest_path: Path, split: str) -> set[tuple[str, int]]:
    payload = read_manifest(manifest_path)
    if "splits" in payload:
        if split == "all":
            records = [
                *payload["splits"].get("train", []),
                *payload["splits"].get("test", []),
            ]
        else:
            records = list(payload["splits"].get(split, []))
    else:
        records = list(payload.get("tasks", []))
    return {(str(item["benchmark"]), int(item["task_id"])) for item in records}


def _trace_from_manifest(
    manifest_path: Path,
    baseline: Optional[str],
    split: str,
    allowed_keys: Optional[set[tuple[str, int]]] = None,
) -> Optional[str]:
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = json.load(fh)

    if manifest.get("score_status") == "unavailable":
        return None
    if baseline and _canonical_or_raw(manifest.get("method", "")) != baseline:
        return None
    task_id = manifest.get("task_id")
    if allowed_keys is not None:
        benchmark = manifest.get("benchmark")
        if benchmark is None or task_id is None:
            return None
        if (str(benchmark), int(task_id)) not in allowed_keys:
            return None
    elif split != "all" and (task_id is None or not _in_split(task_id, split)):
        return None

    trace_path = manifest_path.with_name("memory_trace.jsonl")
    return str(trace_path.resolve()) if trace_path.is_file() else None


def discover_sft_traces(
    paths: Iterable[str],
    baseline: Optional[str] = None,
    split: str = "train",
    manifest: Optional[str] = None,
) -> List[str]:
    """Resolve direct traces and benchmark manifests into usable SFT traces."""
    if split not in ("all", "train", "test", "train_hard", "test_hard"):
        raise ValueError("split must be one of: all, train, test, train_hard, test_hard")
    canonical = canonical_baseline(baseline) if baseline else None
    allowed_keys = _manifest_keys(Path(manifest), split) if manifest else None
    traces = set()

    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            manifests = sorted(path.rglob("summary.json"))
        elif path.name == "summary.json":
            manifests = [path]
        elif path.suffix == ".jsonl":
            manifest = path.with_name("summary.json")
            if not manifest.is_file():
                traces.add(str(path.resolve()))
                continue
            manifests = [manifest]
        else:
            continue

        for manifest in manifests:
            trace_path = _trace_from_manifest(
                manifest, canonical, split, allowed_keys=allowed_keys
            )
            if trace_path:
                traces.add(trace_path)
    return sorted(traces)


def load_samples(trace_paths: List[str]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for path in trace_paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if ev.get("event") != "memory_decision" or not ev.get("proposal"):
                    continue
                target = ev.get("target") or {}
                vis = target.get("visibility")
                recipients = target.get("target_recipients") or ()
                act = canonical_action(vis, recipients)
                if not target.get("exists", True) and act != "absent":
                    act = "absent"
                label = serialize_action(act)
                p = ev["proposal"]
                proposal = MemoryProposal(
                    proposal_id=p["proposal_id"], task_id=p["task_id"],
                    agent_id=p["agent_id"], source=p.get("source", "worker"),
                    title=p.get("title") or "", raw_value=p.get("raw_value") or "",
                    step_index=p.get("step_index", 0),
                )
                samples.append({
                    "feat": features(proposal, []),
                    "label": label,
                    "proposal": proposal,
                })
    return samples


def accuracy(policy: LocalPolicyController, samples: List[Dict[str, Any]]) -> float:
    if not samples:
        return 0.0
    classes = sorted(set(s["label"] for s in samples) | {"absent", "global", "targeted"})
    hits = 0
    for s in samples:
        scores = policy.scores(s["feat"], s.get("proposal"))
        if max(classes, key=lambda c: scores.get(c, 0.0)) == s["label"]:
            hits += 1
    return hits / len(samples)


def train(trace_paths: List[str], out_path: str, epochs: int = 20,
          lr: float = 0.05) -> Dict[str, Any]:
    from collections import Counter
    samples = load_samples(trace_paths)
    policy = LocalPolicyController()
    before = accuracy(policy, samples)
    if samples:
        classes = sorted(set(s["label"] for s in samples) | {"absent", "global", "targeted"})
        counts = Counter(s["label"] for s in samples)
        total = len(samples)
        class_weights = {
            v: total / (len(classes) * max(counts[v], 1))
            for v in classes
        }
        for _ in range(epochs):
            for s in samples:
                w_c = class_weights.get(s["label"], 1.0)
                policy.update(s["feat"], s["label"], lr=lr * w_c, candidate_actions=classes)
    after = accuracy(policy, samples)
    policy.save(out_path)
    stats = {"samples": len(samples), "accuracy_before": before,
             "accuracy_after": after, "out": out_path}
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a controller from decision traces.")
    parser.add_argument("--mode", choices=("linear_sft", "linear_rl", "qwen_sft", "qwen_rl", "sft", "rl"),
                        default="linear_sft",
                        help="linear_sft/linear_rl (or sft/rl): linear LocalPolicy; "
                             "qwen_sft: LoRA SFT; "
                             "qwen_rl: trajectory-level GRPO (Chapter 4)")
    parser.add_argument("--traces", nargs="+", default=None,
                        help="trace files or summary.json manifests")
    parser.add_argument("--run-dir", action="append", default=[],
                        help="benchmark run directory to search for summaries")
    parser.add_argument("--baseline", default=None,
                        help="optional source baseline; legacy aliases are accepted")
    parser.add_argument("--split", choices=("all", "train", "test", "train_hard", "test_hard"), default="train",
                        help="episode split for manifest-backed traces")
    parser.add_argument("--manifest", default=None,
                        help="frozen experiment manifest for summary-backed trace filtering")
    parser.add_argument("--out", default="runs/local_policy.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--init", default=None, help="SFT checkpoint to start RL from")
    parser.add_argument("--r-episode", type=float, default=1.0,
                        help="fallback episode reward when --rewards is omitted")
    parser.add_argument("--rewards", nargs="+", default=None,
                        help="summary.json paths (parallel to --traces) providing real "
                             "task_score for the RL credit loop")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B",
                        help="base model for Qwen LoRA training")
    parser.add_argument("--outcome-gated", action="store_true", default=True,
                        help="RLVR outcome-gating on auxiliary collaboration bonuses")
    parser.add_argument("--target-card-budget", type=int, default=12,
                        help="SimPO card budget threshold for density penalty")
    parser.add_argument("--anchor-coeff", type=float, default=0.005,
                        help="SFT anchor coefficient to prevent covariate drift")
    args = parser.parse_args()
    mode = {"sft": "linear_sft", "rl": "linear_rl"}.get(args.mode, args.mode)
    if mode in ("linear_sft", "qwen_sft"):
        trace_paths = discover_sft_traces(
            [*(args.traces or []), *args.run_dir],
            baseline=args.baseline,
            split=args.split,
            manifest=args.manifest,
        )
        if not trace_paths:
            parser.error("no usable traces found")
    else:
        if not args.traces:
            parser.error("--traces is required for RL")
        trace_paths = args.traces

    if mode == "qwen_sft":
        from marble.controllers import qwen_lora

        pairs = qwen_lora.export_sft_pairs(trace_paths)
        qwen_lora.train_qwen_sft(pairs, args.out, args.base_model, epochs=args.epochs)
    elif mode == "qwen_rl":
        from marble.controllers import qwen_lora

        if not args.rewards:
            parser.error("--rewards is required for qwen_rl")
        rewards = [
            float(json.load(open(reward_path, encoding="utf-8")).get("task_score", 0.0))
            for reward_path in args.rewards
        ]
        epochs = 1 if args.epochs == 20 else args.epochs
        qwen_lora.train_qwen_rl(
            trace_paths, args.out, args.base_model, rewards=rewards,
            epochs=epochs, init_checkpoint=args.init,
        )
    elif mode == "linear_rl":
        from marble.controllers.rl_controller import train_rl

        task_scores = None
        baseline = 0.0
        if args.rewards:
            task_scores = []
            for rp in args.rewards:
                with open(rp, encoding="utf-8") as fh:
                    task_scores.append(float(json.load(fh).get("task_score", 0.0)))
            baseline = sum(task_scores) / len(task_scores) if task_scores else 0.0
        train_rl(
            trace_paths,
            args.out,
            init_checkpoint=args.init,
            epochs=args.epochs,
            lr=0.05,
            r_episode=args.r_episode,
            task_scores=task_scores,
            same_task_baseline=baseline,
            outcome_gated=args.outcome_gated,
            target_card_budget=args.target_card_budget,
            anchor_coeff=args.anchor_coeff,
        )
    else:
        train(trace_paths, args.out, epochs=args.epochs)

