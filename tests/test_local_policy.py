import json
import os

from marble.controllers import LocalPolicyController, features
from marble.memory import GovernedMemory, MemoryBank, MemoryProposal


def _prop(title="shared result", value="x = 1"):
    return MemoryProposal(proposal_id="t:a:1", task_id="t", agent_id="a",
                          source="worker", title=title, raw_value=value,
                          step_index=1)


def test_zero_weights_decide_absent():
    c = LocalPolicyController()
    assert c.decide(_prop(), []).visibility == "absent"
    assert c.decide(_prop(), []).exists is False


def test_sft_updates_learn_shared_signal(tmp_path):
    c = LocalPolicyController()
    samples = [("shared plan for all", "global"), ("private note", "private"),
               ("", "absent")]
    for _ in range(50):
        for title, label in samples:
            value = "v" if title else ""
            c.update(features(_prop(title or "t", value), []), label)
    assert c.decide(_prop("shared plan for all"), []).visibility == "global"
    assert c.decide(_prop("private note"), []).visibility == "private"
    assert c.decide(_prop("anything", ""), []).visibility == "absent"


def test_save_load_roundtrip(tmp_path):
    c = LocalPolicyController()
    path = str(tmp_path / "policy.json")
    c.save(path)
    loaded = LocalPolicyController.load(path)
    assert loaded.weights == c.weights
    with open(path) as fh:
        assert set(json.load(fh).keys()) == {"absent", "private", "global"}


def test_local_policy_decide_supersedes():
    from marble.memory.schema import MemoryItem
    c = LocalPolicyController()
    c.weights["global"]["bias"] = 1.0
    item = MemoryItem(
        memory_id="m1", proposal_id="p1", task_id="t", title="shared result",
        raw_value="old", visibility="global", owner_id=None, source_agent="a",
        source="worker", step_index=0, active=True, supersedes=None, created_at=1,
    )
    prop = _prop(title="shared result", value="new")
    decision = c.decide(prop, [item])
    assert decision.exists is True
    assert decision.visibility == "global"
    assert decision.supersedes == "m1"
