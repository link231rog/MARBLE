"""Loader for MultiAgentBench <bench>_main.jsonl files (spec §12).

Reads only; never modifies source files. Missing optional keys default
(minecraft records omit llm/memory/metrics/output).
"""
from __future__ import annotations

import json
from pathlib import Path

from .task_schema import BenchmarkTask

BENCHMARKS = ("coding", "research", "database", "bargaining", "minecraft")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_path(benchmark: str) -> Path:
    return _REPO_ROOT / "multiagentbench" / benchmark / f"{benchmark}_main.jsonl"


def load_tasks(
    benchmark: str,
    path: str | Path | None = None,
    limit: int | None = None,
    start: int = 0,
    task_ids: list[int] | None = None,
) -> list[BenchmarkTask]:
    """Load BenchmarkTask records, preserving original field values."""
    if benchmark not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark!r}; choose from {BENCHMARKS}")
    jsonl = Path(path) if path else _default_path(benchmark)
    tasks: list[BenchmarkTask] = []
    with open(jsonl, encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tasks.append(
                BenchmarkTask(
                    benchmark=benchmark,
                    # ponytail: minecraft records ship no task_id — fall back to line index
                    task_id=int(r["task_id"]) if "task_id" in r else idx,
                    scenario=str(r.get("scenario", "")),
                    task=str(r["task"]["content"]),
                    agents=tuple(r.get("agents", ())),
                    relationships=tuple(r.get("relationships", ())),
                    environment=dict(r.get("environment", {})),
                    memory=dict(r.get("memory", {})),
                    metrics=dict(r.get("metrics", {})),
                    engine_planner=dict(r.get("engine_planner", {})),
                    output=dict(r.get("output", {})),
                    llm=str(r.get("llm", "")),
                    task_data=dict(r["task"]) if isinstance(r.get("task"), dict) else {"content": str(r.get("task", ""))},
                )
            )
    if task_ids is not None:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.task_id in wanted]
    else:
        tasks = tasks[start:] if start else tasks
        if limit is not None:
            tasks = tasks[:limit]
    return tasks
