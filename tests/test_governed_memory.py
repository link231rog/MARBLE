from pathlib import Path

import pytest

from marble.memory import MemoryBank, MemoryProposal, MemoryTargetState, TraceLogger


def proposal(
    proposal_id: str = "p1",
    task_id: str = "task-1",
    agent_id: str = "agent-1",
) -> MemoryProposal:
    return MemoryProposal(
        proposal_id=proposal_id,
        task_id=task_id,
        agent_id=agent_id,
        source="worker",
        title="Useful result",
        raw_value="result body",
        step_index=1,
    )


def test_private_memory_is_visible_only_to_owner() -> None:
    bank = MemoryBank()
    item = bank.apply(
        proposal(),
        MemoryTargetState(True, "private", owner_id="agent-1"),
    )

    assert item is not None
    assert [card.memory_id for card in bank.visible_keys("agent-1", "task-1")] == ["p1"]
    assert bank.visible_keys("agent-2", "task-1") == []
    assert bank.read("p1", "agent-1", "task-1").raw_value == "result body"
    with pytest.raises(PermissionError):
        bank.read("p1", "agent-2", "task-1")


def test_global_memory_is_visible_to_all_agents() -> None:
    bank = MemoryBank()
    bank.apply(proposal(), MemoryTargetState(True, "global"))

    assert len(bank.visible_keys("agent-1", "task-1")) == 1
    assert len(bank.visible_keys("agent-2", "task-1")) == 1
    assert bank.read("p1", "agent-2", "task-1").raw_value == "result body"


def test_absent_proposal_is_not_stored() -> None:
    bank = MemoryBank()

    assert bank.apply(proposal(), MemoryTargetState(False, "absent")) is None
    assert bank.all_items() == []


def test_task_scope_is_enforced() -> None:
    bank = MemoryBank()
    bank.apply(proposal(task_id="task-1"), MemoryTargetState(True, "global"))

    assert bank.visible_keys("agent-2", "task-2") == []
    with pytest.raises(KeyError):
        bank.read("p1", "agent-2", "task-2")


def test_supersede_deactivates_old_item() -> None:
    bank = MemoryBank()
    bank.apply(proposal(), MemoryTargetState(True, "global"))
    replacement = proposal(proposal_id="p2")

    bank.apply(
        replacement,
        MemoryTargetState(True, "global", supersedes="p1"),
    )

    assert bank.get("p1") is not None
    assert bank.get("p1").active is False
    assert [card.memory_id for card in bank.visible_keys("agent-2", "task-1")] == ["p2"]


def test_trace_logger_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "memory" / "events.jsonl"
    logger = TraceLogger(str(path))
    item = proposal()
    target = MemoryTargetState(True, "private", owner_id="agent-1")

    logger.log_proposal(item)
    logger.log_decision(item, target)
    logger.log_read("p1", "agent-1", "task-1")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert '"event": "memory_read"' in lines[-1]
