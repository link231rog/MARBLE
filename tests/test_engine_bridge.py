import json

from marble.controllers import GlobalAlwaysController, PrivateOnlyController
from marble.experiments.engine_bridge import (
    MemoryStep,
    build_governed_agent_cls,
    build_governed_engine_cls,
)
from marble.memory.rewards import token_count
from marble.memory import GovernedMemory, MemoryBank, TraceLogger


def _memory(tmp_path, controller=None, trace=True):
    trace_path = tmp_path / "trace.jsonl"
    mem = GovernedMemory(
        MemoryBank(),
        controller or GlobalAlwaysController(),
        trace=TraceLogger(str(trace_path)) if trace else None,
    )
    return mem, trace_path


def _events(trace_path):
    with open(trace_path) as fh:
        return [json.loads(line) for line in fh]


def _store_global(mem, title="shared result", value="answer is 42"):
    from marble.memory.schema import MemoryProposal

    item = mem.submit(
        MemoryProposal(
            proposal_id=f"t:seed:{title}", task_id="t", agent_id="seed", source="worker",
            title=title, raw_value=value, step_index=0,
        )
    )
    assert item is not None
    return item


def test_before_act_no_memory_returns_task_unchanged():
    step = MemoryStep(None)
    assert step.before_act("a1", "do thing") == "do thing"
    assert step.after_act("a1", "out") is None


def test_before_act_lists_keys_and_reads_selected(tmp_path):
    mem, tp = _memory(tmp_path)
    item = _store_global(mem)
    step = MemoryStep(mem, selector_fn=lambda p: json.dumps({"memory_ids": [item.memory_id]}))
    step.task_id = "t"
    augmented = step.before_act("coder", "write code")
    assert "Shared memory keys:" in augmented and item.title in augmented
    assert "Read notes:" in augmented and "answer is 42" in augmented
    assert step.reads_this_episode == 1
    events = _events(tp)
    assert any(e["event"] == "memory_read" and e["reader_id"] == "coder" for e in events)
    exposure = next(e for e in events if e["event"] == "memory_exposure")
    assert exposure["memory_ids"] == [item.memory_id]


def test_before_act_counts_key_cards_and_notes_separately(tmp_path):
    mem, _ = _memory(tmp_path)
    item = _store_global(mem, title="shared key", value="read this note")
    step = MemoryStep(mem, selector_fn=lambda p: json.dumps({"memory_ids": [item.memory_id]}))
    step.task_id = "t"

    step.before_act("coder", "task")

    context = step._context_by_agent["coder"]
    key_text = (
        "Shared memory keys:\n"
        f"- [M1] {item.title} ({item.visibility})"
    )
    notes_text = f"- [{item.title}] read this note"
    assert context["memory_card_tokens"] == token_count(key_text)
    assert context["injected_memory_tokens"] == token_count(notes_text)


def test_selector_cap_and_unknown_ids_skipped(tmp_path):
    mem, _ = _memory(tmp_path)
    i1 = _store_global(mem, "a")
    i2 = _store_global(mem, "b")
    i3 = _store_global(mem, "c")
    # asks for 3 incl unknown; capped to max_reads_per_step=2
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


def test_invalid_selection_json_rejected(tmp_path):
    mem, _ = _memory(tmp_path)
    _store_global(mem)
    step = MemoryStep(mem, selector_fn=lambda p: "garbage")
    step.task_id = "t"
    augmented = step.before_act("coder", "q")
    assert "Read notes:" not in augmented
    assert len(step.selection_rejections) == 1


def test_private_card_of_other_agent_not_offered(tmp_path):
    mem, _ = _memory(tmp_path, controller=PrivateOnlyController())
    from marble.memory.schema import MemoryProposal

    mem.submit(MemoryProposal(proposal_id="p1", task_id="t", agent_id="other",
                              source="worker", title="secret plan", raw_value="hush",
                              step_index=0))
    step = MemoryStep(mem, selector_fn=lambda p: json.dumps({"memory_ids": ["anything"]}))
    step.task_id = "t"
    augmented = step.before_act("coder", "q")
    assert "Shared memory keys:" not in augmented


def test_after_act_submits_proposal_and_steps_increment(tmp_path):
    mem, tp = _memory(tmp_path)
    step = MemoryStep(mem)
    step.task_id = "t"
    step.before_act("coder", "q")  # real lifecycle calls before_act each turn
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


def test_governed_agent_feeds_augmented_task_to_base(tmp_path):
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


def test_governed_engine_cls_overrides_agents_only():
    class DummyEnv:
        pass

    class DummyEngine:
        def __init__(self, config):
            self.config = config
            self.environment = DummyEnv()
            self.agents = self._initialize_agents(config.agents)

        def _initialize_agents(self, agent_configs):
            raise AssertionError("parent should be overridden")

    class FakeAgent:
        def __init__(self, config, env, model):
            self.agent_id = config["agent_id"]
            self.governed = None

    GovernedEngine = build_governed_engine_cls(DummyEngine, agent_cls=FakeAgent)
    from types import SimpleNamespace

    engine = GovernedEngine(
        SimpleNamespace(llm="fake", agents=[{"agent_id": "a1"}, {"agent_id": "a2"}])
    )
    assert [a.agent_id for a in engine.agents] == ["a1", "a2"]
    assert all(a.governed is None for a in engine.agents)  # runner attaches harness later


def test_governed_engine_attaches_harness_before_init(tmp_path):
    # regression for FATAL: agents read harness during __init__, so it must be
    # set on the class BEFORE Engine(config) is constructed.
    mem, _ = _memory(tmp_path)
    harness = MemoryStep(mem, max_cards=6, max_reads_per_step=2, selector="top")

    class DummyEnv:
        pass

    class DummyEngine:
        def __init__(self, config):
            self.config = config
            self.environment = DummyEnv()
            self.agents = self._initialize_agents(config.agents)

        def _initialize_agents(self, agent_configs):
            agents = []
            for ac in agent_configs:
                a = FakeAgent(ac, self.environment, self.config.llm)
                a.governed = self.memory_harness  # mirrors GovernedEngine override
                agents.append(a)
            return agents

    class FakeAgent:
        def __init__(self, config, env, model):
            self.agent_id = config["agent_id"]
            self.governed = None

    GovernedEngine = build_governed_engine_cls(DummyEngine, agent_cls=FakeAgent)
    GovernedEngine.memory_harness = harness  # set BEFORE constructing engine
    from types import SimpleNamespace

    engine = GovernedEngine(
        SimpleNamespace(llm="fake", agents=[{"agent_id": "a1"}])
    )
    assert engine.agents[0].governed is harness  # attached, not None


def test_selector_top_reads_top_ranked_cards(tmp_path):
    mem, trace_path = _memory(tmp_path)
    item = _store_global(mem, title="first", value="alpha")
    item2 = _store_global(mem, title="second", value="beta")
    harness = MemoryStep(
        mem, max_cards=6, max_reads_per_step=1, selector="top"
    )
    harness.task_id = "t"
    text = harness.before_act("reader", "t")
    assert "alpha" in text and "beta" not in text  # top-1 read, capped
    assert harness.reads_this_episode == 1
    assert harness.reads_this_episode == 1
    events = [
        e for e in _events(trace_path)
        if e.get("event") == "memory_read"
    ]
    assert events and events[0]["memory_id"] == item.memory_id
