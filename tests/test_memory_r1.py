import pytest

from marble.memory.memory_r1 import MemoryR1Memory
from marble.memory.schema import MemoryProposal


def proposal(
    proposal_id: str = "p1",
    task_id: str = "task-1",
    title: str = "Database schema fix",
    raw_value: str = "Use a shared schema.",
) -> MemoryProposal:
    return MemoryProposal(
        proposal_id=proposal_id,
        task_id=task_id,
        agent_id="agent-1",
        source="worker",
        title=title,
        raw_value=raw_value,
        step_index=1,
    )


def test_add_creates_active_global_item_and_trace() -> None:
    memory = MemoryR1Memory()

    result = memory.add(proposal())

    assert result.item is not None
    assert result.item.active is True
    assert result.trace.operation == "ADD"
    assert result.trace.memory_id == "p1"
    assert result.trace.task_id == "task-1"
    assert result.trace.title == "Database schema fix"
    assert result.trace.raw_value == "Use a shared schema."


def test_update_replaces_value_and_preserves_identity() -> None:
    memory = MemoryR1Memory()
    memory.add(proposal())

    result = memory.update(
        "p1",
        proposal(
            proposal_id="p2",
            title="Database schema correction",
            raw_value="Use the corrected shared schema.",
        ),
    )

    assert result.trace.operation == "UPDATE"
    assert result.item is not None
    assert result.item.memory_id == "p1"
    assert result.item.title == "Database schema correction"
    assert result.item.raw_value == "Use the corrected shared schema."
    assert result.item.created_at == 1
    assert result.item.updated_at == 2


def test_delete_deactivates_item_and_emits_trace() -> None:
    memory = MemoryR1Memory()
    memory.add(proposal())

    result = memory.delete("p1", "task-1")

    assert result.trace.operation == "DELETE"
    assert result.item is not None
    assert result.item.active is False
    assert memory.get("p1") is not None
    assert memory.get("p1").active is False
    with pytest.raises(KeyError):
        memory.read("p1", "task-1")


def test_noop_emits_trace_without_modifying_store() -> None:
    memory = MemoryR1Memory()
    memory.add(proposal())

    result = memory.noop("task-1", "p1")

    assert result.item is None
    assert result.trace.operation == "NOOP"
    assert result.trace.memory_id == "p1"
    assert result.trace.task_id == "task-1"
    assert memory.read("p1", "task-1").raw_value == "Use a shared schema."


def test_task_scope_is_enforced_for_crud_and_reads() -> None:
    memory = MemoryR1Memory()
    memory.add(proposal(task_id="task-1"))

    assert memory.active_items("task-2") == []
    with pytest.raises(KeyError):
        memory.update("p1", proposal(task_id="task-2"))
    with pytest.raises(KeyError):
        memory.delete("p1", "task-2")
    with pytest.raises(KeyError):
        memory.read("p1", "task-2")


def test_retrieval_is_deterministic_and_returns_only_active_task_cards() -> None:
    memory = MemoryR1Memory()
    memory.add(proposal("p1", title="Database schema fix"))
    memory.add(proposal("p2", title="Team schema decision"))
    memory.add(proposal("p3", title="Unrelated note"))
    memory.add(proposal("p4", task_id="task-2", title="Database task two"))
    memory.delete("p3", "task-1")

    ranked = memory.retrieve("task-1", query="database schema", top_k=3)

    assert [card.memory_id for card in ranked] == ["p1", "p2"]
    assert [card.title for card in ranked] == [
        "Database schema fix",
        "Team schema decision",
    ]
