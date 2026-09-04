import json

from marble.memory import MemoryProposal, MemoryR1Adapter, TraceLogger
from marble.memory.memory_r1_adapter import parse_crud_decision


def proposal(proposal_id="p1", value="shared result", title="Team decision"):
    return MemoryProposal(
        proposal_id=proposal_id,
        task_id="task-1",
        agent_id="agent-1",
        source="worker",
        title=title,
        raw_value=value,
        step_index=1,
    )


def test_adapter_maps_new_and_duplicate_outputs_to_add_and_update(tmp_path):
    path = tmp_path / "trace.jsonl"
    runtime = MemoryR1Adapter(trace=TraceLogger(str(path)))

    first = runtime.submit(proposal())
    second = runtime.submit(
        proposal("p2", value="corrected result", title="Team decision")
    )

    assert first.memory_id == second.memory_id == "p1"
    assert runtime.read("p1", "agent-2", "task-1").raw_value == "corrected result"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [
        event["operation"]
        for event in events
        if event["event"] == "memory_r1_operation"
    ] == ["ADD", "UPDATE"]


def test_adapter_is_global_and_task_scoped():
    runtime = MemoryR1Adapter()
    runtime.submit(proposal())

    assert [card.visibility for card in runtime.visible_keys("agent-2", "task-1")] == [
        "global"
    ]
    assert runtime.visible_keys("agent-2", "task-2") == []


def test_adapter_applies_manager_delete_and_distills_reads(tmp_path):
    calls = []
    runtime = MemoryR1Adapter(
        manager=lambda proposal, active: {
            "operation": "ADD" if not active else "DELETE",
            "memory_id": active[0].memory_id if active else None,
        },
        distill_fn=lambda task, notes: calls.append((task, notes)) or "compact evidence",
    )
    assert runtime.submit(proposal()) is not None
    assert runtime.submit(proposal("p2", value="invalidate the old note")) is None

    assert runtime.visible_keys("agent-2", "task-1") == []
    assert runtime.distill("solve task", ["a", "b"]) == ["compact evidence"]
    assert calls == [("solve task", ["a", "b"])]


def test_crud_parser_rejects_invalid_or_inactive_targets():
    item = MemoryR1Adapter().memory.add(proposal()).item
    assert item is not None
    active = [item]

    assert parse_crud_decision("not json", active)["operation"] == "NOOP"
    assert parse_crud_decision(
        '{"operation":"DELETE","memory_id":"missing"}', active
    )["operation"] == "NOOP"
    assert parse_crud_decision(
        '{"operation":"UPDATE","memory_id":"p1"}', active
    ) == {"operation": "UPDATE", "memory_id": "p1"}
    # Memory-R1 Appendix C.1 formats
    assert parse_crud_decision(
        '{"event":"NONE","id":"0"}', active
    ) == {"operation": "NOOP", "memory_id": None}
    assert parse_crud_decision(
        '{"event":"UPDATE","id":"p1","text":"updated"}', active
    ) == {"operation": "UPDATE", "memory_id": "p1"}
    assert parse_crud_decision(
        '{"memory":[{"id":"0","event":"NONE"},{"id":"1","text":"fact","event":"ADD"}]}', active
    ) == {"operation": "ADD", "memory_id": None}
    assert parse_crud_decision(
        '{"memory":[{"id":"p1","text":"updated","event":"UPDATE","old_memory":"old"}]}', active
    ) == {"operation": "UPDATE", "memory_id": "p1"}

