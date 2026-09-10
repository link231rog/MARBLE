"""Deterministic task manifests for the frozen MultiAgentBench experiment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def task_record(task: Any) -> dict[str, Any]:
    return {"benchmark": task.benchmark, "task_id": int(task.task_id), "agent_count": len(task.agents)}


def write_manifest(path: str | Path, tasks: Iterable[Any], *, seed: int, setting: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"seed": seed, "setting": setting, "tasks": [task_record(t) for t in tasks]}, indent=2) + "\n", encoding="utf-8")


def read_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must contain an object")
    if "tasks" not in payload and "splits" not in payload:
        raise ValueError("manifest must contain tasks or splits")
    return payload


def select_manifest_tasks(tasks: Iterable[Any], path: str | Path) -> list[Any]:
    payload = read_manifest(path)
    records = payload.get("tasks", [])
    for split_records in payload.get("splits", {}).values():
        records.extend(split_records)
    keys = {(str(item["benchmark"]), int(item["task_id"])) for item in records}
    return [task for task in tasks if (task.benchmark, int(task.task_id)) in keys]


def build_frozen_manifest(tasks: Iterable[Any], *, seed: int = 42) -> tuple[list[Any], list[Any]]:
    """Return six training and eighteen test tasks.

    Database contributes 3 train and 9 test tasks (12 total, all 5-agent).
    Research contributes 3 train and 9 test tasks (12 total), with its tasks
    stratified across 3, 4, and 5 agents (1 train and 3 test tasks per agent count).
    ``seed`` is retained in the API for manifest metadata; selection is ID-stable
    once the manifest is checked in.
    """
    del seed
    grouped: dict[tuple[str, int], list[Any]] = {}
    for task in tasks:
        count = len(task.agents)
        if task.benchmark in {"database", "research"} and 3 <= count <= 5:
            grouped.setdefault((task.benchmark, count), []).append(task)
    for values in grouped.values():
        values.sort(key=lambda task: int(task.task_id))

    train: list[Any] = []
    test: list[Any] = []
    database = grouped.get(("database", 5), [])
    if len(database) < 12:
        raise ValueError("frozen manifest needs at least 12 eligible database tasks")
    train.extend(database[:3])
    test.extend(database[3:12])
    for count in (3, 4, 5):
        values = grouped.get(("research", count), [])
        if len(values) < 4:
            raise ValueError(
                f"frozen manifest needs at least 4 research tasks with {count} agents"
            )
        train.append(values[0])
        test.extend(values[1:4])
    return sorted(train, key=lambda task: (task.benchmark, int(task.task_id))), sorted(
        test, key=lambda task: (task.benchmark, int(task.task_id))
    )


def load_manifest_tasks(path: str | Path, *, split: str = "all") -> list[Any]:
    """Load only the benchmark tasks named by a frozen manifest."""
    payload = read_manifest(path)
    valid_splits = {
        "all",
        "train",
        "test",
        "train_standard",
        "train_hard",
        "test_standard",
        "test_hard",
    }
    if split not in valid_splits:
        raise ValueError(f"manifest split must be one of {sorted(valid_splits)}")
    records: list[dict[str, Any]] = []
    if "splits" in payload:
        splits_dict = payload["splits"]
        if split == "all":
            for s_name in ("train", "test", "train_standard", "train_hard", "test_standard", "test_hard"):
                for item in splits_dict.get(s_name, []):
                    if item not in records:
                        records.append(item)
        elif split in splits_dict:
            records = list(splits_dict[split])
        elif split == "train":
            records = list(splits_dict.get("train", [])) or (
                list(splits_dict.get("train_standard", [])) + list(splits_dict.get("train_hard", []))
            )
        elif split == "test":
            records = list(splits_dict.get("test", [])) or (
                list(splits_dict.get("test_standard", [])) + list(splits_dict.get("test_hard", []))
            )
    else:
        records = list(payload.get("tasks", []))
    from marble.benchmarks import load_tasks

    by_benchmark: dict[str, list[int]] = {}
    for record in records:
        by_benchmark.setdefault(str(record["benchmark"]), []).append(
            int(record["task_id"])
        )
        cnt = int(record.get("agent_count", 3))
        if cnt < 1 or cnt > 30:
            raise ValueError(f"manifest contains an unsupported agent count: {cnt}")
    result: list[Any] = []
    for benchmark in sorted(by_benchmark):
        result.extend(load_tasks(benchmark, task_ids=by_benchmark[benchmark]))
    result.sort(key=lambda task: (task.benchmark, int(task.task_id)))
    expected = {(str(r["benchmark"]), int(r["task_id"])) for r in records}
    actual = {(task.benchmark, int(task.task_id)) for task in result}
    if actual != expected:
        raise ValueError("manifest references missing or duplicated benchmark tasks")
    expected_counts = {
        (str(record["benchmark"]), int(record["task_id"])): int(
            record.get("agent_count", 0)
        )
        for record in records
    }
    if any(
        len(task.agents) != expected_counts[(task.benchmark, task.task_id)]
        for task in result
    ):
        raise ValueError("manifest agent_count does not match benchmark data")
    return result
