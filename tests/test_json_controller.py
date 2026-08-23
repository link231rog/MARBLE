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
    assert (target.exists, target.visibility, target.owner_id, target.supersedes) == (
        True, "global", None, None)
    assert ctrl.rejections == []


def test_private_supersedes_resolved_by_title():
    bank = MemoryBank()
    mem = GovernedMemory(bank, JsonController(_llm({"visibility": "private", "supersedes": None})))
    first = mem.submit(_proposal("p1"))
    assert first is not None
    ctrl = JsonController(_llm({"visibility": "private", "supersedes": "shared result"}))
    items = [it for it in bank.all_items() if it.active]
    target = ctrl.decide(_proposal("p2"), items)
    assert target.supersedes == first.memory_id
    assert target.owner_id == "coder"


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


def test_prompt_follows_four_block_contract():
    seen = {}
    def llm(prompt):
        seen["prompt"] = prompt
        return '{"visibility": "global", "supersedes": null}'
    ctrl = JsonController(llm, max_value_chars=20,
                          agent_capabilities=("coding",), task_goal="solve fizzbuzz")
    ctrl.decide(_proposal(value="x" * 100), [])
    p = seen["prompt"]
    # four frozen input blocks present
    for block in ("[TASK]", "[AGENT]", "[PROPOSAL]", "[ACTIVE MEMORY INDEX]"):
        assert block in p
    assert "task_goal: solve fizzbuzz" in p
    assert "agent_capabilities: ['coding']" in p
    # contract-excluded fields must NOT leak into the prompt
    for forbidden in ("task_id:", "step_index:", "proposal_id:", "agent_id:"):
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
