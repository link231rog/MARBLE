import json
from pathlib import Path

import pytest

from marble.memory import (
    GovernedMemory,
    MemoryBank,
    MemoryProposal,
    MemoryTargetState,
    TraceLogger,
)


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


def test_targeted_memory_is_visible_only_to_recipients() -> None:
    bank = MemoryBank()
    item = bank.apply(
        proposal(),
        MemoryTargetState(True, "targeted", target_recipients=("agent-1",)),
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
    target = MemoryTargetState(True, "targeted", target_recipients=("agent-1",))

    logger.log_proposal(item)
    logger.log_decision(item, target)
    logger.log_read("p1", "agent-1", "task-1")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert '"event": "memory_read"' in lines[-1]


def test_submit_logs_current_controller_decision_context(tmp_path: Path) -> None:
    class Controller:
        def decide(self, item, current_state):
            self.last_prompt = f"prompt:{item.proposal_id}"
            self.last_raw = f"raw:{item.proposal_id}"
            return MemoryTargetState(True, "global")

    path = tmp_path / "events.jsonl"
    memory = GovernedMemory(
        MemoryBank(),
        Controller(),
        trace=TraceLogger(str(path)),
    )
    memory.submit(proposal("p1"), controller_prompt="stale", controller_output="stale")
    memory.submit(proposal("p2"), controller_prompt="stale", controller_output="stale")

    decisions = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "memory_decision"
    ]

    assert decisions[0]["controller_prompt"] == "prompt:p1"
    assert decisions[0]["controller_output"] == "raw:p1"
    assert decisions[0]["active_memory_index"] == []
    assert decisions[1]["controller_prompt"] == "prompt:p2"
    assert decisions[1]["controller_output"] == "raw:p2"
    assert [item["memory_id"] for item in decisions[1]["active_memory_index"]] == ["p1"]


def test_cross_agent_injection_producer_no_implicit_access() -> None:
    """Agent-1 proposes a memory routed strictly to Agent-2. Agent-1 has NO implicit access."""
    bank = MemoryBank()
    p = proposal(proposal_id="p_shared", agent_id="agent-1")
    target = MemoryTargetState(True, "targeted", target_recipients=("agent-2",))
    item = bank.apply(p, target)

    assert item is not None
    assert item.target_recipients == ("agent-2",)
    assert item.source_agent == "agent-1"

    # Agent-2 sees it and can read it
    agent2_keys = [c.memory_id for c in bank.visible_keys("agent-2", "task-1")]
    assert "p_shared" in agent2_keys
    assert bank.read("p_shared", "agent-2", "task-1").raw_value == "result body"

    # Producer (agent-1) has NO implicit access: not in target_recipients -> PermissionError
    agent1_keys = [c.memory_id for c in bank.visible_keys("agent-1", "task-1")]
    assert "p_shared" not in agent1_keys
    with pytest.raises(PermissionError):
        bank.read("p_shared", "agent-1", "task-1")

    # Agent-3 has zero access and receives PermissionError
    agent3_keys = [c.memory_id for c in bank.visible_keys("agent-3", "task-1")]
    assert "p_shared" not in agent3_keys
    with pytest.raises(PermissionError):
        bank.read("p_shared", "agent-3", "task-1")


def test_multi_agent_targeted_sharing() -> None:
    """Memory targeted to multiple agents: all listed agents have access, unlisted do not."""
    bank = MemoryBank()
    p = proposal(proposal_id="p_multi", agent_id="agent-1")
    target = MemoryTargetState.targeted(recipients=["agent-1", "agent-2"])
    item = bank.apply(p, target)

    assert item is not None
    assert item.visibility == "targeted"
    assert item.target_recipients == ("agent-1", "agent-2")

    # Both agent-1 and agent-2 can see and read it
    assert "p_multi" in [c.memory_id for c in bank.visible_keys("agent-1", "task-1")]
    assert bank.read("p_multi", "agent-1", "task-1").raw_value == "result body"

    assert "p_multi" in [c.memory_id for c in bank.visible_keys("agent-2", "task-1")]
    assert bank.read("p_multi", "agent-2", "task-1").raw_value == "result body"

    # Unlisted third-party (agent-3) is completely shielded
    assert "p_multi" not in [c.memory_id for c in bank.visible_keys("agent-3", "task-1")]
    with pytest.raises(PermissionError):
        bank.read("p_multi", "agent-3", "task-1")
