"""Frozen task schema shared by all benchmark loaders (spec §12)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkTask:
    """One immutable benchmark record; raw values preserved as loaded."""

    benchmark: str
    task_id: int
    scenario: str
    task: str  # task["content"]
    agents: tuple = ()
    relationships: tuple = ()
    environment: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    engine_planner: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)
    llm: str = ""  # model string for workers (e.g. '', 'gpt-4o-mini')
