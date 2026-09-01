"""Fresh-rollout Qwen controller training loop.

The worker model, memory engine, retriever, and reward function stay frozen.
Only the local Qwen LoRA controller is sampled and updated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from marble.benchmarks import load_tasks
from marble.controllers.qwen_lora import train_qwen_rl
from marble.experiments.run_benchmark import run_task


def _read_score(summary_path: Path) -> float:
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "task_score", 0.0
        )
        return float(value or 0.0)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def collect_rollouts(
    tasks: List[Any],
    checkpoint: str,
    out_root: Path,
    *,
    rollouts_per_task: int,
    qwen_base_model: str,
    qwen_temperature: float,
    max_iterations: Optional[int],
    max_reads_per_step: int,
    retrieval: str,
    worker_model: Optional[str],
    ablation: Optional[str],
) -> tuple[List[str], List[float], List[Dict[str, Any]]]:
    """Collect independent episodes from the current controller checkpoint."""
    if rollouts_per_task < 2:
        raise ValueError("rollouts_per_task must be at least 2 for same-task baselines")
    if qwen_temperature <= 0:
        raise ValueError("fresh rollout requires --qwen-temperature > 0")

    traces: List[str] = []
    rewards: List[float] = []
    manifest: List[Dict[str, Any]] = []
    for task in tasks:
        for rollout_id in range(rollouts_per_task):
            episode_root = out_root / f"rollout-{rollout_id:03d}"
            summary = run_task(
                task,
                "qwen_sft",
                episode_root,
                max_iterations=max_iterations,
                max_reads_per_step=max_reads_per_step,
                retriever=retrieval,
                llm=worker_model or "",
                ablation=ablation,
                controller_checkpoint=checkpoint,
                qwen_base_model=qwen_base_model,
                qwen_temperature=qwen_temperature,
            )
            trace_path = (
                episode_root
                / "qwen_sft"
                / task.benchmark
                / str(task.task_id)
                / "memory_trace.jsonl"
            )
            summary_path = trace_path.with_name("summary.json")
            score = _read_score(summary_path)
            record = {
                "benchmark": task.benchmark,
                "task_id": task.task_id,
                "rollout_id": rollout_id,
                "status": summary.get("status"),
                "task_score": score,
                "trace": str(trace_path.resolve()),
                "summary": str(summary_path.resolve()),
            }
            manifest.append(record)
            if trace_path.exists():
                traces.append(str(trace_path.resolve()))
                rewards.append(score)
    return traces, rewards, manifest


def train_fresh_rollouts(
    benchmark: str,
    checkpoint: str,
    out_root: str,
    *,
    task_ids: Optional[List[int]] = None,
    limit: Optional[int] = None,
    rollouts_per_task: int = 2,
    rounds: int = 1,
    qwen_base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    qwen_temperature: float = 0.7,
    max_iterations: Optional[int] = None,
    max_reads_per_step: int = 2,
    retrieval: str = "key_first",
    worker_model: Optional[str] = None,
    ablation: Optional[str] = None,
) -> Dict[str, Any]:
    """Run repeated fresh-rollout -> REINFORCE updates from an SFT adapter."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    tasks = load_tasks(benchmark, task_ids=task_ids, limit=limit)
    if not tasks:
        raise ValueError("no benchmark tasks selected")

    root = Path(out_root)
    current_checkpoint = str(Path(checkpoint).resolve())
    round_records: List[Dict[str, Any]] = []
    for round_id in range(rounds):
        round_root = root / f"round-{round_id:03d}"
        traces, rewards, manifest = collect_rollouts(
            tasks,
            current_checkpoint,
            round_root,
            rollouts_per_task=rollouts_per_task,
            qwen_base_model=qwen_base_model,
            qwen_temperature=qwen_temperature,
            max_iterations=max_iterations,
            max_reads_per_step=max_reads_per_step,
            retrieval=retrieval,
            worker_model=worker_model,
            ablation=ablation,
        )
        (round_root / "rollout_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if len(traces) < 2:
            raise ValueError("fresh rollout produced fewer than two trace files")
        next_checkpoint = root / f"checkpoint-{round_id + 1:03d}"
        train_qwen_rl(
            traces,
            str(next_checkpoint),
            qwen_base_model,
            rewards=rewards,
            init_checkpoint=current_checkpoint,
        )
        round_record = {
            "round": round_id,
            "input_checkpoint": current_checkpoint,
            "output_checkpoint": str(next_checkpoint.resolve()),
            "traces": len(traces),
            "rewards": rewards,
            "manifest": str((round_root / "rollout_manifest.json").resolve()),
        }
        round_records.append(round_record)
        current_checkpoint = str(next_checkpoint.resolve())

    result = {
        "benchmark": benchmark,
        "task_ids": [task.task_id for task in tasks],
        "rollouts_per_task": rollouts_per_task,
        "rounds": rounds,
        "final_checkpoint": current_checkpoint,
        "rounds_detail": round_records,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "training_manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fresh-rollout Qwen LoRA policy-gradient training"
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--checkpoint", required=True, help="initial SFT LoRA adapter")
    parser.add_argument("--out", required=True)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rollouts-per-task", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--qwen-base-model", default="Qwen/Qwen3-4B-Instruct-2507"
    )
    parser.add_argument("--qwen-temperature", type=float, default=0.7)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-reads-per-step", type=int, default=2)
    parser.add_argument("--retrieval", default="key_first")
    parser.add_argument("--worker-model", default=None)
    parser.add_argument("--ablation", default=None)
    args = parser.parse_args(argv)
    task_ids = [int(value) for value in args.task_ids.split(",") if value.strip()]
    train_fresh_rollouts(
        args.benchmark,
        args.checkpoint,
        args.out,
        task_ids=task_ids or None,
        limit=args.limit,
        rollouts_per_task=args.rollouts_per_task,
        rounds=args.rounds,
        qwen_base_model=args.qwen_base_model,
        qwen_temperature=args.qwen_temperature,
        max_iterations=args.max_iterations,
        max_reads_per_step=args.max_reads_per_step,
        retrieval=args.retrieval,
        worker_model=args.worker_model,
        ablation=args.ablation,
    )


if __name__ == "__main__":
    main()
