"""Fresh-rollout Qwen controller training loop.

The worker model, memory engine, retriever, and reward function stay frozen.
Only the local Qwen LoRA controller is sampled and updated.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from marble.benchmarks import load_tasks
from marble.controllers.qwen_lora import train_qwen_rl
from marble.experiments.run_benchmark import _apply_split, run_task


def _read_score(summary_path: Path) -> float:
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8")).get(
            "task_score", 0.0
        )
        return float(value or 0.0)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def _checkpoint_complete(path: Path) -> bool:
    """Return whether an adapter directory has the files needed to resume."""
    if not path.is_dir() or not (path / "adapter_config.json").is_file():
        return False
    return any(
        (path / filename).is_file()
        for filename in (
            "adapter_model.safetensors",
            "adapter_model.bin",
            "pytorch_model.bin",
        )
    )


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp_path = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _append_training_event(path: Path, event: str, **fields: Any) -> None:
    record = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _load_training_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _round_record(
    round_id: int,
    input_checkpoint: str,
    output_checkpoint: Path,
    round_root: Path,
    traces: int = 0,
    rewards: Optional[List[float]] = None,
) -> Dict[str, Any]:
    return {
        "round": round_id,
        "input_checkpoint": input_checkpoint,
        "output_checkpoint": str(output_checkpoint.resolve()),
        "traces": traces,
        "rewards": rewards or [],
        "manifest": str((round_root / "rollout_manifest.json").resolve()),
    }


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
    split: str = "train",
    max_cards: Optional[int] = None,
    _lambda: Optional[float] = None,
    beta: Optional[float] = None,
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
                "ours_sft",
                episode_root,
                max_iterations=max_iterations,
                max_reads_per_step=max_reads_per_step,
                retriever=retrieval,
                llm=worker_model or "",
                ablation=ablation,
                controller_checkpoint=checkpoint,
                qwen_base_model=qwen_base_model,
                qwen_temperature=qwen_temperature,
                max_cards=max_cards if max_cards is not None else 6,
                lambda_=_lambda if _lambda is not None else 0.05,
                beta=beta if beta is not None else 0.25,
            )
            # Prefer canonical ours_sft directory, fall back to legacy qwen_sft
            ours_trace = (
                episode_root
                / "ours_sft"
                / task.benchmark
                / str(task.task_id)
                / "memory_trace.jsonl"
            )
            qwen_trace = (
                episode_root
                / "qwen_sft"
                / task.benchmark
                / str(task.task_id)
                / "memory_trace.jsonl"
            )
            trace_path = ours_trace if ours_trace.exists() else qwen_trace
            summary_path = trace_path.with_name("summary.json")
            score = _read_score(summary_path)
            # Reject or skip a rollout unless summary score_status equals available
            # and status is successful
            if summary.get("score_status") != "available" or summary.get("status") != "ok":
                continue
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
    split: str = "train",
    max_cards: Optional[int] = None,
    _lambda: Optional[float] = None,
    beta: Optional[float] = None,
) -> Dict[str, Any]:
    """Run repeated fresh-rollout -> REINFORCE updates from an SFT adapter."""
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    tasks = load_tasks(benchmark, task_ids=task_ids, limit=limit)
    tasks = _apply_split(tasks, split)
    if not tasks:
        raise ValueError("no benchmark tasks selected")

    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    event_path = root / "training_events.jsonl"
    training_manifest_path = root / "training_manifest.json"
    previous = _load_training_manifest(training_manifest_path)
    previous_records = {
        int(record["round"]): record
        for record in previous.get("rounds_detail", [])
        if isinstance(record, dict) and str(record.get("round", "")).isdigit()
    }
    current_checkpoint = str(Path(checkpoint).resolve())
    round_records: List[Dict[str, Any]] = []
    _append_training_event(
        event_path,
        "start",
        benchmark=benchmark,
        rounds=rounds,
        task_ids=[task.task_id for task in tasks],
        input_checkpoint=current_checkpoint,
    )
    result: Dict[str, Any] = {
        "benchmark": benchmark,
        "task_ids": [task.task_id for task in tasks],
        "rollouts_per_task": rollouts_per_task,
        "rounds": rounds,
        "final_checkpoint": current_checkpoint,
        "rounds_detail": round_records,
        "status": "running",
    }
    _atomic_write_json(training_manifest_path, result)
    try:
        for round_id in range(rounds):
            round_root = root / f"round-{round_id:03d}"
            round_manifest_path = round_root / "rollout_manifest.json"
            previous_record = previous_records.get(round_id, {})
            recorded_output = previous_record.get("output_checkpoint")
            next_checkpoint = Path(
                recorded_output
                if recorded_output
                else root / f"checkpoint-{round_id + 1:03d}"
            )
            if (
                round_manifest_path.is_file()
                and _checkpoint_complete(next_checkpoint)
            ):
                round_record = dict(previous_record) if previous_record else _round_record(
                    round_id,
                    current_checkpoint,
                    next_checkpoint,
                    round_root,
                )
                round_record["round"] = round_id
                round_record["output_checkpoint"] = str(next_checkpoint.resolve())
                round_records.append(round_record)
                current_checkpoint = str(next_checkpoint.resolve())
                _append_training_event(
                    event_path,
                    "round_skip",
                    round=round_id,
                    output_checkpoint=current_checkpoint,
                    reason="rollout_manifest_and_checkpoint_complete",
                )
                continue

            _append_training_event(
                event_path,
                "round_start",
                round=round_id,
                input_checkpoint=current_checkpoint,
            )
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
                split=split,
                max_cards=max_cards,
                _lambda=_lambda,
                beta=beta,
            )
            _atomic_write_json(
                round_manifest_path,
                {"round": round_id, "rollouts": manifest},
            )
            if len(traces) < 2:
                raise ValueError("fresh rollout produced fewer than two trace files")
            train_qwen_rl(
                traces,
                str(next_checkpoint),
                qwen_base_model,
                rewards=rewards,
                init_checkpoint=current_checkpoint,
            )
            if not _checkpoint_complete(next_checkpoint):
                raise RuntimeError(
                    f"training did not produce a complete checkpoint: {next_checkpoint}"
                )
            round_record = _round_record(
                round_id,
                current_checkpoint,
                next_checkpoint,
                round_root,
                traces=len(traces),
                rewards=rewards,
            )
            round_records.append(round_record)
            current_checkpoint = str(next_checkpoint.resolve())
            _append_training_event(
                event_path,
                "round_end",
                round=round_id,
                output_checkpoint=current_checkpoint,
                traces=len(traces),
            )
            result.update(
                final_checkpoint=current_checkpoint,
                rounds_detail=round_records,
            )
            _atomic_write_json(training_manifest_path, result)

        result.update(
            final_checkpoint=current_checkpoint,
            rounds_detail=round_records,
            status="completed",
        )
        _atomic_write_json(training_manifest_path, result)
        _append_training_event(
            event_path, "end", status="completed", final_checkpoint=current_checkpoint
        )
    except Exception as exc:
        result.update(
            final_checkpoint=current_checkpoint,
            rounds_detail=round_records,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        _atomic_write_json(training_manifest_path, result)
        _append_training_event(
            event_path,
            "error",
            error=f"{type(exc).__name__}: {exc}",
            round=len(round_records),
        )
        _append_training_event(
            event_path, "end", status="failed", final_checkpoint=current_checkpoint
        )
        raise
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
    parser.add_argument("--split", default="train", help="train/test split for tasks")
    parser.add_argument("--max-cards", type=int, default=None, help="max cards for retrieval")
    parser.add_argument("--lambda", type=float, default=None, dest="_lambda",
                        help="lambda value for reward")
    parser.add_argument("--beta", type=float, default=None, help="beta value for reward")
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
        split=args.split,
        max_cards=args.max_cards,
        _lambda=args._lambda,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
