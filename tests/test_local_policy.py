import json
from marble.controllers import LocalPolicyController, features
from marble.memory import MemoryProposal
from marble.memory.schema import MemoryItem


def _prop(title="shared result", value="x = 1"):
    return MemoryProposal(proposal_id="t:a:1", task_id="t", agent_id="a",
                          source="worker", title=title, raw_value=value,
                          step_index=1)


def test_local_policy_learning_and_persistence(tmp_path):
    c = LocalPolicyController()
    assert c.decide(_prop(), []).visibility == "absent"

    # SFT updates
    samples = [("shared plan for all", "global"), ("private note", "targeted"), ("", "absent")]
    for _ in range(50):
        for title, label in samples:
            value = "v" if title else ""
            c.update(features(_prop(title or "t", value), []), label)
    assert c.decide(_prop("shared plan for all"), []).visibility == "global"
    assert c.decide(_prop("private note"), []).visibility == "targeted"
    assert c.decide(_prop("anything", ""), []).visibility == "absent"

    # Save / load roundtrip
    path = str(tmp_path / "policy.json")
    c.save(path)
    loaded = LocalPolicyController.load(path)
    assert loaded.weights == c.weights
    with open(path) as fh:
        assert set(json.load(fh).keys()) == {"absent", "targeted", "global"}


def test_local_policy_supersedes():
    c = LocalPolicyController()
    c.weights["global"]["bias"] = 1.0
    item = MemoryItem(
        memory_id="m1", proposal_id="p1", task_id="t", title="shared result",
        raw_value="old", visibility="global", source_agent="a",
        source="worker", step_index=0, active=True, supersedes=None, created_at=1,
    )
    prop = _prop(title="shared result", value="new")
    decision = c.decide(prop, [item])
    assert decision.visibility == "global"
    assert decision.supersedes == "m1"

    # Placeholder title must not supersede
    item_placeholder = MemoryItem(
        memory_id="m2", proposal_id="p2", task_id="t", title="<concise summary key>",
        raw_value="old", visibility="global", source_agent="a",
        source="worker", step_index=0, active=True, supersedes=None, created_at=1,
    )
    prop_placeholder = _prop(title="<concise summary key>", value="new")
    decision_ph = c.decide(prop_placeholder, [item_placeholder])
    assert decision_ph.supersedes is None


def test_local_policy_multi_agent_routing():
    c = LocalPolicyController()
    c.weights["targeted"]["bias"] = 1.0
    c.set_context(
        task_goal="Fix database deadlock and optimize query",
        agent_role_map={"db_specialist": "database engineer", "frontend": "ui designer"},
    )
    prop = MemoryProposal(
        proposal_id="p1", task_id="t1", agent_id="frontend",
        source="worker", title="slow query on pg_stat", raw_value="SELECT * FROM pg_stat_activity",
        step_index=1,
    )
    decision = c.decide(prop, [])
    assert decision.visibility == "targeted"
    assert decision.target_recipients == ("db_specialist",)

    # Multi-agent recipient weights learning
    c2 = LocalPolicyController()
    c2.set_context(
        task_goal="Collaborative feature",
        agent_role_map={"agent_1": "worker 1", "agent_2": "worker 2", "agent_3": "worker 3"},
    )
    feat = features(_prop(title="joint sync", value="state exchange"), [])
    for _ in range(30):
        c2.update(feat, ("agent_1", "agent_2"), lr=0.1)

    prop2 = _prop(title="joint sync", value="state exchange")
    decision2 = c2.decide(prop2, [])
    assert decision2.exists is True
    assert decision2.visibility == "targeted"
    assert decision2.target_recipients == ("agent_1", "agent_2")

    # Candidate recipient sets
    c_five = LocalPolicyController()
    c_five.set_context(
        task_goal="5 agents",
        agent_role_map={f"agent_{i}": f"role_{i}" for i in range(5)},
    )
    candidates = c_five.candidate_recipient_sets(_prop())
    assert len(candidates) == 31
    assert ("agent_0", "agent_1") in candidates
