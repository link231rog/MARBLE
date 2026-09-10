import json
from types import SimpleNamespace
from marble.controllers import GlobalAlwaysController, PrivateOnlyController
from marble.experiments.engine_bridge import (
    MemoryStep,
    build_governed_agent_cls,
    build_governed_engine_cls,
)
from marble.memory.rewards import token_count
from marble.memory import GovernedMemory, MemoryBank, TraceLogger
from marble.memory.schema import MemoryProposal


def _memory(tmp_path, controller=None, trace=True):
    trace_path = tmp_path / "trace.jsonl"
    mem = GovernedMemory(
        MemoryBank(),
        controller or GlobalAlwaysController(),
        trace=TraceLogger(str(trace_path)) if trace else None,
    )
    return mem, trace_path


def _store_global(mem, title="shared result", value="answer is 42"):
    item = mem.submit(
        MemoryProposal(
            proposal_id=f"t:seed:{title}", task_id="t", agent_id="seed", source="worker",
            title=title, raw_value=value, step_index=0,
        )
    )
    assert item is not None
    return item


def _events(tp):
    with open(tp) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_memory_step_before_act_lifecycle(tmp_path):
    # No memory case
    step0 = MemoryStep(None)
    assert step0.before_act("a1", "do thing") == "do thing"
    assert step0.after_act("a1", "out") is None

    # Normal before_act with selection
    mem, tp = _memory(tmp_path)
    item = _store_global(mem, title="shared key", value="read this note")
    step = MemoryStep(mem, selector_fn=lambda p: json.dumps({"memory_ids": [item.memory_id]}))
    step.task_id = "t"
    augmented = step.before_act("coder", "write code")
    assert "Shared memory keys:" in augmented and item.title in augmented
    assert "Read notes:" in augmented and "read this note" in augmented
    assert step.reads_this_episode == 1

    events = _events(tp)
    assert any(e["event"] == "memory_read" and e["reader_id"] == "coder" for e in events)
    exposure = next(e for e in events if e["event"] == "memory_exposure")
    assert exposure["memory_ids"] == [item.memory_id]

    context = step._context_by_agent["coder"]
    key_text = f"Shared memory keys:\n- [M1] {item.title} ({item.visibility})"
    notes_text = f"- [{item.title}] read this note"
    assert context["memory_card_tokens"] == token_count(key_text)
    assert context["injected_memory_tokens"] == token_count(notes_text)


def test_memory_step_selector_rules(tmp_path):
    mem, _ = _memory(tmp_path)
    i1 = _store_global(mem, "a")
    i2 = _store_global(mem, "b")
    i3 = _store_global(mem, "c")

    # Selector cap & unknown id skip
    step = MemoryStep(
        mem,
        selector_fn=lambda p: json.dumps({"memory_ids": [i1.memory_id, "ghost", i2.memory_id, i3.memory_id]}),
        max_reads_per_step=2,
    )
    step.task_id = "t"
    out = step.before_act("coder", "q")
    assert "Read notes:" in out
    read_section = out.split("Read notes:\n")[1]
    assert read_section.count("- [") == 2
    assert step.reads_this_episode == 2

    # Invalid selection json rejected
    step_err = MemoryStep(mem, selector_fn=lambda p: "garbage")
    step_err.task_id = "t"
    assert "Read notes:" not in step_err.before_act("coder", "q")
    assert len(step_err.selection_rejections) == 1

    # Private card of other agent not offered
    priv_dir = tmp_path / "priv"
    priv_dir.mkdir(parents=True)
    mem_priv, _ = _memory(priv_dir, controller=PrivateOnlyController())
    mem_priv.submit(MemoryProposal(proposal_id="p1", task_id="t", agent_id="other", source="worker", title="secret", raw_value="hush", step_index=0))
    step_priv = MemoryStep(mem_priv, selector_fn=lambda p: json.dumps({"memory_ids": ["anything"]}))
    step_priv.task_id = "t"
    assert "Shared memory keys:" not in step_priv.before_act("coder", "q")


def test_after_act_proposal_lifecycle(tmp_path):
    mem, tp = _memory(tmp_path)
    step = MemoryStep(mem)
    step.task_id = "t"
    step.before_act("coder", "q")
    mid = step.after_act("coder", "Useful result: done")
    assert mid is not None
    step.before_act("coder", "q2")
    mid2 = step.after_act("coder", "Second result")
    assert mid2 is not None and mid2 != mid
    assert step.steps["coder"] == 2
    events = _events(tp)
    decisions = [e for e in events if e["event"] == "memory_decision"]
    assert len(decisions) == 2
    assert decisions[0]["proposal"]["step_index"] == 1


def test_governed_agent_and_engine_integration(tmp_path):
    calls = []
    class DummyBase:
        def __init__(self, config=None, env=None, model=None):
            self.agent_id = config["agent_id"]
        def act(self, task):
            calls.append(task)
            return ("out", None)

    mem, _ = _memory(tmp_path)
    item = _store_global(mem)
    GovernedAgent = build_governed_agent_cls(DummyBase)
    harness = MemoryStep(mem, selector_fn=lambda p: json.dumps({"memory_ids": []}))
    harness.task_id = "t"
    agent = GovernedAgent(config={"agent_id": "a1"})
    agent.governed = harness
    agent.act("task text")
    assert "task text" in calls[0]
    assert "Shared memory keys:" in calls[0] and item.title in calls[0]

    # Engine integration
    class FakeAgent:
        def __init__(self, config, env, model):
            self.agent_id = config["agent_id"]
            self.governed = None

    class DummyEngine:
        def __init__(self, config):
            self.config = config
            self.agents = [FakeAgent(ac, None, config.llm) for ac in config.agents]

    GovernedEngine = build_governed_engine_cls(DummyEngine, agent_cls=FakeAgent)
    GovernedEngine.memory_harness = harness
    engine = GovernedEngine(SimpleNamespace(llm="fake", agents=[{"agent_id": "a1"}, {"agent_id": "a2"}]))
    assert [a.agent_id for a in engine.agents] == ["a1", "a2"]


def test_selector_top_and_budget_limits(tmp_path):
    mem, trace_path = _memory(tmp_path)
    _store_global(mem, title="first", value="alpha")
    _store_global(mem, title="second", value="beta")
    harness = MemoryStep(mem, max_cards=6, max_reads_per_step=1, selector="top")
    harness.task_id = "t"
    text = harness.before_act("reader", "t")
    assert "alpha" in text and "beta" not in text
    assert harness.reads_this_episode == 1

    # Global add all budget enforcement
    m2_dir = tmp_path / "m2"
    m2_dir.mkdir(parents=True)
    mem2, trace_path2 = _memory(m2_dir)
    for i in range(10):
        _store_global(mem2, title=f"item_{i}", value=f"val_{i}")
    harness2 = MemoryStep(mem2, max_cards=5, max_reads_per_step=2, selector="top", baseline="global_add_all")
    harness2.task_id = "t"
    harness2.before_act("reader", "run task")
    assert harness2.reads_this_episode == 2
    events = [e for e in _events(trace_path2) if e.get("event") == "memory_read"]
    assert len(events) == 2
    exposure = next(e for e in _events(trace_path2) if e.get("event") == "memory_exposure")
    assert len(exposure["memory_ids"]) == 5
