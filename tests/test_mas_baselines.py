import pytest
from marble.experiments.baselines import MAIN_BASELINES, baseline_spec, canonical_baseline
from marble.memory.mas_baselines import CollabMemAdapter, COPPERAdapter, GMemoryAdapter
from marble.memory.schema import MemoryProposal
from marble.memory.trace import TraceLogger


def _prop(pid: str, agent: str, title: str, val: str) -> MemoryProposal:
    return MemoryProposal(
        proposal_id=pid,
        task_id="t1",
        agent_id=agent,
        source="worker",
        title=title,
        raw_value=val,
        step_index=1,
    )


def test_mas_baselines_registered():
    for name in ("g_memory_style", "collabmem_style", "copper_style"):
        assert name in MAIN_BASELINES
        spec = baseline_spec(name)
        assert spec.uses_memory is True
        assert spec.requires_crud_manager is True


def test_g_memory_hierarchical_tiers(tmp_path):
    trace = TraceLogger(str(tmp_path / "gmem_trace.jsonl"))
    adapter = GMemoryAdapter(trace=trace)

    p1 = _prop("p1", "agent1", "Investigation Task", "investigate vacuum root cause")
    p2 = _prop("p2", "agent1", "Root cause is VACUUM", "confirmed VACUUM FULL blocking tables")
    p3 = _prop("p3", "agent2", "raw query log", "select * from pg_stat_activity")

    m1 = adapter.submit(p1)
    m2 = adapter.submit(p2)
    m3 = adapter.submit(p3)

    assert m1.tier == "query"
    assert m2.tier == "insight"
    assert m3.tier == "interaction"

    # Insights must be prioritized at the top of visible keys
    keys = adapter.visible_keys("agent3", "t1", top_k=3)
    assert len(keys) == 3
    assert "[INSIGHT]" in keys[0].title

    read_item = adapter.read(m2.memory_id, "agent3", "t1")
    assert read_item.title == "Root cause is VACUUM"


def test_collabmem_bipartite_access_control(tmp_path):
    trace = TraceLogger(str(tmp_path / "collab_trace.jsonl"))
    adapter = CollabMemAdapter(trace=trace)

    # 1. Private note: only agent1 can access
    p_priv = _prop("priv1", "agent1", "my scratchpad", "testing local query")
    adapter.submit(p_priv)

    # 2. Group note: agent1 mentions agent2
    p_grp = _prop("grp1", "agent1", "hand off to agent2", "collaborate with agent2 on locks")
    adapter.submit(p_grp)

    # 3. Public note: final decision
    p_pub = _prop("pub1", "agent1", "final decision", "root cause confirmed as LOCK_CONTENTION")
    adapter.submit(p_pub)

    # agent3 is neither author nor in group -> should ONLY see public
    keys_agent3 = adapter.visible_keys("agent3", "t1")
    assert len(keys_agent3) == 1
    assert "[PUBLIC]" in keys_agent3[0].title

    # agent2 is in group -> should see public and group, but NOT private
    keys_agent2 = adapter.visible_keys("agent2", "t1")
    assert len(keys_agent2) == 2
    assert any("[GROUP]" in k.title for k in keys_agent2)
    assert any("[PUBLIC]" in k.title for k in keys_agent2)

    # agent1 (author) should see all 3
    keys_agent1 = adapter.visible_keys("agent1", "t1")
    assert len(keys_agent1) == 3


def test_copper_counterfactual_reflections(tmp_path):
    trace = TraceLogger(str(tmp_path / "copper_trace.jsonl"))
    adapter = COPPERAdapter(trace=trace)

    # Failed query generates counterfactual reflection
    p_fail = _prop("f1", "agent1", "query error", "column does not exist in table")
    adapter.submit(p_fail)

    all_items = adapter.bank.all_items()
    # base item + counterfactual reflection item
    assert len(all_items) == 2
    refl = [it for it in all_items if it.is_reflection]
    assert len(refl) == 1
    assert "[Reflect: Avoid]" in refl[0].title

    # Reflections are boosted to top
    keys = adapter.visible_keys("agent2", "t1", top_k=2)
    assert "[Reflect: Avoid]" in keys[0].title
