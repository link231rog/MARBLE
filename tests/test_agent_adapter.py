from pathlib import Path

import pytest

from marble.experiments.baselines import BASELINES, make_memory
from marble.experiments.rollout import run_mock_episode
from marble.memory.adapter import MemoryAwareAgentAdapter, format_cards


OUTPUTS_A = ["Alpha found the schema fix details"]
OUTPUTS_B = ["Beta wrote a local scratch note"]


def test_adapter_roundtrip_stores_and_reads() -> None:
    memory = make_memory("global_always")
    adapter = MemoryAwareAgentAdapter(object(), memory)

    cards = adapter.before_step("task-1", "agent-1")
    assert cards == []

    memory_id = adapter.after_step(OUTPUTS_A[0])
    assert memory_id == "task-1:agent-1:1"

    other = MemoryAwareAgentAdapter(object(), memory)
    visible = other.before_step("task-1", "agent-2")
    assert [card.memory_id for card in visible] == [memory_id]
    assert other.read(memory_id) == OUTPUTS_A[0]


def test_adapter_requires_before_step_first() -> None:
    adapter = MemoryAwareAgentAdapter(object(), make_memory("heuristic"))

    with pytest.raises(RuntimeError):
        adapter.after_step("orphan output")


def test_absent_baseline_returns_no_memory_id() -> None:
    adapter = MemoryAwareAgentAdapter(object(), make_memory("no_memory"))
    adapter.before_step("task-1", "agent-1")

    assert adapter.after_step(OUTPUTS_A[0]) is None


def test_private_memory_hidden_cross_agent_via_adapter() -> None:
    memory = make_memory("private_only")
    writer = MemoryAwareAgentAdapter(object(), memory)
    writer.before_step("task-1", "agent-1")
    memory_id = writer.after_step(OUTPUTS_A[0])
    assert memory_id is not None

    reader = MemoryAwareAgentAdapter(object(), memory)
    assert reader.before_step("task-1", "agent-2") == []
    with pytest.raises(PermissionError):
        reader.read(memory_id)


def test_run_mock_episode_smoke_all_baselines(tmp_path: Path) -> None:
    plan = {"agent-2": []}
    for baseline in BASELINES:
        trace = str(tmp_path / f"{baseline}.jsonl")
        stats = run_mock_episode(
            make_memory(baseline, trace_path=trace),
            "task-1",
            {"agent-1": OUTPUTS_A, "agent-2": OUTPUTS_B},
            read_plan=plan,
        )

        expected_stored = 0 if baseline == "no_memory" else 2
        assert stats["proposals_stored"] == expected_stored
        assert Path(trace).exists()


def test_format_cards_renders_key_fields() -> None:
    from marble.memory import MemoryCard

    text = format_cards(
        [
            MemoryCard(
                memory_id="m1",
                task_id="t",
                title="Schema fix",
                visibility="global",
            )
        ]
    )
    assert text == "- [global] m1: Schema fix"
    assert format_cards([]) == "(no memories)"
