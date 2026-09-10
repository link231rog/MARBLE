import json

import pytest

from marble.controllers import JsonController
from marble.memory import GovernedMemory, MemoryBank, MemoryProposal


def _proposal(pid="p1", title="shared result", value="the answer is 42"):
    return MemoryProposal(
        proposal_id=pid, task_id="t", agent_id="coder", source="worker",
        title=title, raw_value=value, step_index=1,
    )


def _llm(payload):
    return lambda prompt: payload if isinstance(payload, str) else json.dumps(payload)


def test_valid_global_decision():
    ctrl = JsonController(_llm({"visibility": "global", "supersedes": None}))
    target = ctrl.decide(_proposal(), [])
    assert (target.exists, target.visibility, target.supersedes) == (
        True, "global", None)
    assert ctrl.rejections == []


def test_private_supersedes_resolved_by_memory_id():
    bank = MemoryBank()
    role_map = {"coder": "developer"}
    mem = GovernedMemory(bank, JsonController(_llm({"visibility": ["coder"], "supersedes": None}), agent_role_map=role_map))
    first = mem.submit(_proposal("p1"))
    assert first is not None
    # review P2: supersedes resolves by exact memory_id only, not by title
    ctrl = JsonController(_llm({"visibility": ["coder"], "supersedes": first.memory_id}), agent_role_map=role_map)
    items = [it for it in bank.all_items() if it.active]
    target = ctrl.decide(_proposal("p2"), items)
    assert target.supersedes == first.memory_id
    assert target.target_recipients == ("coder",)


def test_supersedes_by_title_is_rejected():
    bank = MemoryBank()
    role_map = {"coder": "developer"}
    mem = GovernedMemory(bank, JsonController(_llm({"visibility": ["coder"], "supersedes": None}), agent_role_map=role_map))
    mem.submit(_proposal("p1", title="shared result"))
    ctrl = JsonController(_llm({"visibility": ["coder"], "supersedes": "shared result"}), agent_role_map=role_map)
    items = [it for it in bank.all_items() if it.active]
    target = ctrl.decide(_proposal("p2"), items)
    assert target.visibility == "absent"  # title no longer resolves
    assert "not found" in ctrl.rejections[-1]["reason"]


def test_invalid_json_rejected_and_recorded():
    ctrl = JsonController(_llm("not json at all"))
    target = ctrl.decide(_proposal(), [])
    assert (target.exists, target.visibility) == (False, "absent")
    assert len(ctrl.rejections) == 1
    assert "not valid JSON" in ctrl.rejections[0]["reason"]


@pytest.mark.parametrize("payload,reason", [
    ('{"visibility": "pending", "supersedes": null}', "invalid visibility"),
    ('{"visibility": "global"}', "expected exactly keys"),
    ('{"visibility": "absent", "supersedes": "m1"}', "null supersedes"),
])
def test_schema_violations_rejected(payload, reason):
    ctrl = JsonController(_llm(payload))
    target = ctrl.decide(_proposal(), [])
    assert target.visibility == "absent"
    assert reason in ctrl.rejections[0]["reason"]


def test_unknown_supersedes_target_rejected():
    ctrl = JsonController(_llm({"visibility": "global", "supersedes": "ghost-id"}))
    target = ctrl.decide(_proposal(), [])
    assert target.visibility == "absent"
    assert "not found" in ctrl.rejections[0]["reason"]


def test_prompt_follows_frozen_system_and_three_block_contract():
    seen = {}
    def llm(prompt):
        seen["prompt"] = prompt
        return '{"visibility": "global", "supersedes": null}'
    ctrl = JsonController(
        llm,
        max_value_chars=20,
        agent_role_map={"coder": "coding"},
        task_goal="solve fizzbuzz",
    )
    ctrl.decide(_proposal(value="x" * 100), [])
    p = seen["prompt"]
    for block in ("[SYSTEM]", "[TASK]", "[PROPOSAL]", "[ACTIVE MEMORY INDEX]"):
        assert block in p
    assert "task_goal: solve fizzbuzz" in p
    assert 'agent_role_map: {"coder": "coding"}' in p
    assert "agent_reference: coder" in p
    # contract-excluded fields must NOT leak into the prompt
    for forbidden in ("task_id:", "step_index:", "proposal_id:", "agent_capabilities:"):
        assert forbidden not in p
    # value truncation honored
    assert len(p.split("value: ", 1)[1].split("\n")[0]) <= 20


def test_input_ablation_hides_active_memory_index():
    seen = {}
    def llm(prompt):
        seen["prompt"] = prompt
        return '{"visibility": "global", "supersedes": null}'
    ctrl = JsonController(llm, drop_fields=("active_memory_index",))
    target = ctrl.decide(_proposal(), [])
    assert "[ACTIVE MEMORY INDEX]" not in seen["prompt"]
    # ablation forces supersedes null even if model asks for it
    assert target.supersedes is None


def test_visibility_set_routing():
    role_map = {"agent_0": "Coder", "agent_1": "Reviewer", "agent_2": "Manager"}
    ctrl = JsonController(
        _llm({"visibility": ["agent_1"], "supersedes": None}),
        agent_role_map=role_map,
    )
    prop = MemoryProposal(
        proposal_id="p_cross", task_id="t", agent_id="agent_0", source="worker",
        title="review feedback for reviewer", raw_value="Check line 42", step_index=1,
    )
    target = ctrl.decide(prop, [])
    assert target.exists is True
    assert target.visibility == "targeted"
    assert target.target_recipients == ("agent_1",)

    # Apply to bank
    bank = MemoryBank()
    item = bank.apply(prop, target)
    assert item.target_recipients == ("agent_1",)
    assert item.source_agent == "agent_0"

    # agent_1 (in recipient list) can see it
    visible_agent1 = [c.memory_id for c in bank.visible_keys(reader_id="agent_1", task_id="t")]
    assert item.memory_id in visible_agent1

    # agent_0 (producer, but NOT in recipient list) CANNOT see it
    visible_agent0 = [c.memory_id for c in bank.visible_keys(reader_id="agent_0", task_id="t")]
    assert item.memory_id not in visible_agent0

    # agent_2 (unrelated) CANNOT see it
    visible_agent2 = [c.memory_id for c in bank.visible_keys(reader_id="agent_2", task_id="t")]
    assert item.memory_id not in visible_agent2


def test_multi_agent_visibility_routing():
    role_map = {"agent_0": "Coder", "agent_1": "Reviewer", "agent_2": "Manager"}
    ctrl = JsonController(
        _llm({"visibility": ["agent_0", "agent_1"], "supersedes": None}),
        agent_role_map=role_map,
    )
    prop = MemoryProposal(
        proposal_id="p_multi", task_id="t", agent_id="agent_0", source="worker",
        title="joint discussion", raw_value="Check line 42", step_index=1,
    )
    target = ctrl.decide(prop, [])
    assert target.exists is True
    assert target.visibility == "targeted"
    assert target.target_recipients == ("agent_0", "agent_1")

    bank = MemoryBank()
    item = bank.apply(prop, target)
    # Both agent_0 and agent_1 can see it
    assert "p_multi" in [c.memory_id for c in bank.visible_keys(reader_id="agent_0", task_id="t")]
    assert "p_multi" in [c.memory_id for c in bank.visible_keys(reader_id="agent_1", task_id="t")]
    # agent_2 cannot see it
    assert "p_multi" not in [c.memory_id for c in bank.visible_keys(reader_id="agent_2", task_id="t")]

