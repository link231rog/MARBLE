from pathlib import Path

import pytest

from marble.controllers import (
    AbsentController,
    GlobalAlwaysController,
    HeuristicController,
    PrivateOnlyController,
)
from marble.memory import (
    GovernedMemory,
    KeyRetriever,
    MemoryBank,
    MemoryCard,
    MemoryProposal,
    MemoryTargetState,
    TraceLogger,
)


def proposal(
    proposal_id: str = "p1",
    task_id: str = "task-1",
    agent_id: str = "agent-1",
    title: str = "Useful result",
    raw_value: str = "result body",
) -> MemoryProposal:
    return MemoryProposal(
        proposal_id=proposal_id,
        task_id=task_id,
        agent_id=agent_id,
        source="worker",
        title=title,
        raw_value=raw_value,
        step_index=1,
    )


def card(memory_id: str, title: str) -> MemoryCard:
    return MemoryCard(
        memory_id=memory_id,
        task_id="task-1",
        title=title,
        visibility="global",
        owner_id=None,
    )


def test_global_always_controller_admits_everything() -> None:
    target = GlobalAlwaysController().decide(proposal(), [])

    assert target.exists is True
    assert target.visibility == "global"
    assert target.owner_id is None
    assert target.supersedes is None


def test_absent_controller_rejects_everything() -> None:
    target = AbsentController().decide(proposal(), [])

    assert target.exists is False
    assert target.visibility == "absent"


def test_private_only_controller_owns_by_agent() -> None:
    target = PrivateOnlyController().decide(proposal(agent_id="a9"), [])

    assert target.exists is True
    assert target.visibility == "private"
    assert target.owner_id == "a9"


def test_controllers_supersede_duplicate_title() -> None:
    bank = MemoryBank()
    bank.apply(proposal(), MemoryTargetState(True, "global"))
    current_state = [item for item in bank.all_items() if item.active]

    for controller in (GlobalAlwaysController(), PrivateOnlyController()):
        target = controller.decide(proposal(proposal_id="p2"), current_state)
        assert target.supersedes == "p1"


def test_heuristic_controller_rules() -> None:
    controller = HeuristicController()

    absent = controller.decide(proposal(raw_value="   "), [])
    assert (absent.exists, absent.visibility) == (False, "absent")

    shared = controller.decide(proposal(title="Team decision on schema"), [])
    assert (shared.exists, shared.visibility) == (True, "global")

    private = controller.decide(proposal(agent_id="a3", title="Local scratch"), [])
    assert (private.exists, private.visibility, private.owner_id) == (
        True,
        "private",
        "a3",
    )


def test_retriever_ranks_and_cuts() -> None:
    cards = [
        card("c1", "Database schema fix"),
        card("c2", "Shared team decision"),
        card("c3", "Unrelated note"),
    ]

    ranked = KeyRetriever().rank(cards, query="database schema", top_k=2)
    assert [item.memory_id for item in ranked] == ["c1", "c2"]

    no_query = KeyRetriever().rank(cards, query=None, top_k=6)
    assert [item.memory_id for item in no_query] == ["c1", "c2", "c3"]


def test_governed_memory_end_to_end(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace" / "events.jsonl"
    memory = GovernedMemory(
        bank=MemoryBank(),
        controller=HeuristicController(),
        trace=TraceLogger(str(trace_path)),
    )

    first = memory.submit(proposal(title="Team decision on schema"))
    assert first is not None

    second = memory.submit(
        proposal(proposal_id="p2", agent_id="agent-2", title="Private scratch")
    )
    assert second is not None

    visible_owner = memory.visible_keys("agent-1", "task-1", query="decision")
    visible_other = memory.visible_keys("agent-2", "task-1")
    assert {entry.memory_id for entry in visible_owner} >= {"p1"}
    # p1 was stored global by the heuristic, so it is visible to every reader.
    assert {entry.memory_id for entry in visible_other} == {"p1", "p2"}

    with pytest.raises(PermissionError):
        memory.read("p2", "agent-1", "task-1")

    read_item = memory.read("p2", "agent-2", "task-1")
    assert read_item.raw_value == "result body"

    text = trace_path.read_text(encoding="utf-8")
    events = [
        line.split('"event": ')[1].split(",")[0].strip('"')
        for line in text.splitlines()
    ]
    assert events.count("memory_proposal") == 2
    assert events.count("memory_decision") == 2
    assert events.count("memory_read") == 1
