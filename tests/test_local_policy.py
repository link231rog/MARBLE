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
    samples = [("shared plan for all", "global"), ("private note", "targeted"),
               ("", "absent")]
    for _ in range(50):
        for title, label in samples:
            value = "v" if title else ""
            c.update(features(_prop(title or "t", value), []), label)
    assert c.decide(_prop("shared plan for all"), []).visibility == "global"
    assert c.decide(_prop("private note"), []).visibility == "targeted"
    assert c.decide(_prop("anything", ""), []).visibility == "absent"


def test_save_load_roundtrip(tmp_path):
    c = LocalPolicyController()
    path = str(tmp_path / "policy.json")
    c.save(path)
    loaded = LocalPolicyController.load(path)
    assert loaded.weights == c.weights
    with open(path) as fh:
        assert set(json.load(fh).keys()) == {"absent", "targeted", "global"}


def test_local_policy_decide_supersedes():
    from marble.memory.schema import MemoryItem
    c = LocalPolicyController()
    c.weights["global"]["bias"] = 1.0
    item = MemoryItem(
        memory_id="m1", proposal_id="p1", task_id="t", title="shared result",
        raw_value="old", visibility="global", source_agent="a",
        source="worker", step_index=0, active=True, supersedes=None, created_at=1,
    )
    prop = _prop(title="shared result", value="new")
    decision = c.decide(prop, [item])
    assert decision.exists is True
    assert decision.visibility == "global"
    assert decision.supersedes == "m1"


def test_non_supersedable_titles_do_not_supersede():
    from marble.memory.schema import MemoryItem
    c = LocalPolicyController()
    c.weights["global"]["bias"] = 1.0
    item = MemoryItem(
        memory_id="m1", proposal_id="p1", task_id="t", title="<concise summary key>",
        raw_value="old", visibility="global", source_agent="a",
        source="worker", step_index=0, active=True, supersedes=None, created_at=1,
    )
    prop = _prop(title="<concise summary key>", value="new")
    decision = c.decide(prop, [item])
    assert decision.exists is True
    assert decision.supersedes is None  # Must NOT supersede placeholder!


def test_local_policy_multi_agent_routing_by_role_and_content():
    c = LocalPolicyController()
    c.weights["targeted"]["bias"] = 1.0  # Encourage targeted decision
    c.set_context(
        task_goal="Fix database deadlock and optimize query",
        agent_role_map={"db_specialist": "database engineer", "frontend": "ui designer"},
    )
    # Proposal from frontend about a database query
    prop = MemoryProposal(
        proposal_id="p1", task_id="t1", agent_id="frontend",
        source="worker", title="slow query on pg_stat", raw_value="SELECT * FROM pg_stat_activity",
        step_index=1,
    )
    decision = c.decide(prop, [])
    assert decision.exists is True
    assert decision.visibility == "targeted"
    # Must target db_specialist, NOT blindly defaulting to self (frontend)!
    assert decision.target_recipients == ("db_specialist",)


def test_local_policy_learns_multi_agent_recipient_weights():
    c = LocalPolicyController()
    c.set_context(
        task_goal="Collaborative feature",
        agent_role_map={"agent_1": "worker 1", "agent_2": "worker 2", "agent_3": "worker 3"},
    )
    # Train policy to target ("agent_1", "agent_2") on joint sync proposals
    feat = features(_prop(title="joint sync", value="state exchange"), [])
    for _ in range(30):
        c.update(feat, ("agent_1", "agent_2"), lr=0.1)

    prop = _prop(title="joint sync", value="state exchange")
    decision = c.decide(prop, [])
    assert decision.exists is True
    assert decision.visibility == "targeted"
    assert decision.target_recipients == ("agent_1", "agent_2")


def test_candidate_recipient_sets_five_agents():
    c = LocalPolicyController()
    c.set_context(
        task_goal="Database deadlock optimization",
        agent_role_map={
            "agent_0": "coordinator",
            "agent_1": "db_specialist",
            "agent_2": "workload_analyzer",
            "agent_3": "index_optimizer",
            "agent_4": "vacuum_manager",
        },
    )
    candidates = c.candidate_recipient_sets(_prop())
    # 5 singletons + 10 pairs + 10 triplets + 5 quads + 1 quintuple = 31 candidate sets
    assert len(candidates) == 31
    multi_agent_sets = [s for s in candidates if len(s) > 1]
    assert len(multi_agent_sets) == 26
    assert ("agent_0", "agent_1") in candidates
    assert ("agent_0", "agent_1", "agent_2", "agent_3", "agent_4") in candidates


def test_candidate_recipient_sets_large_cluster():
    c = LocalPolicyController()
    agents = {f"agent_{i}": f"role_{i}" for i in range(7)}
    c.set_context(task_goal="Research consensus", agent_role_map=agents)
    # Proposal mentions agent_1, agent_3, agent_5 explicitly
    prop = _prop(title="sync review", value="agent_1, agent_3 and agent_5 please review section 2")
    candidates = c.candidate_recipient_sets(prop)
    # 7 singletons + 21 pairs + 1 mentioned triplet = 29
    assert len([s for s in candidates if len(s) == 1]) == 7
    assert len([s for s in candidates if len(s) == 2]) == 21
    assert ("agent_1", "agent_3", "agent_5") in candidates

