import pytest
from marble.memory.classical_baselines import Mem0Adapter, AMemAdapter, MemoryOSAdapter
from marble.memory.schema import MemoryProposal
from marble.experiments.baselines import canonical_baseline, baseline_spec, MAIN_BASELINES
from marble.experiments.run_benchmark import make_memory_runtime


def test_registry_contains_classical_baselines():
    assert "mem0_style" in MAIN_BASELINES
    assert "amem_style" in MAIN_BASELINES
    assert "memoryos_style" in MAIN_BASELINES
    assert canonical_baseline("mem0") == "mem0_style"
    assert canonical_baseline("amem") == "amem_style"
    assert canonical_baseline("memoryos") == "memoryos_style"


def test_mem0_adapter_lifecycle():
    adapter = Mem0Adapter()
    p1 = MemoryProposal(
        task_id="task-1",
        agent_id="agent-a",
        source="worker",
        step_index=1,
        title="Postgres Config",
        raw_value="shared_buffers is set to 128MB",
        proposal_id="p1",
    )
    res1 = adapter.submit(p1)
    assert res1 is not None
    assert res1.title == "Postgres Config"

    # Query
    keys = adapter.visible_keys(reader_id="agent-b", task_id="task-1", query="buffers", top_k=5)
    assert len(keys) == 1
    assert keys[0].title == "Postgres Config"

    # Update
    p2 = MemoryProposal(
        task_id="task-1",
        agent_id="agent-a",
        source="worker",
        step_index=2,
        title="Postgres Config",
        raw_value="shared_buffers is updated to 1GB",
        proposal_id="p2",
    )
    res2 = adapter.submit(p2)
    assert res2 is not None
    read_item = adapter.read(res2.memory_id, reader_id="agent-b", task_id="task-1")
    assert "1GB" in read_item.raw_value


def test_amem_adapter_linking_and_evolution():
    adapter = AMemAdapter(max_links_per_note=2)
    p1 = MemoryProposal(
        task_id="task-1",
        agent_id="agent-a",
        source="worker",
        step_index=1,
        title="Lock Contention",
        raw_value="High row lock contention on orders table detected",
        proposal_id="p1",
    )
    adapter.submit(p1)

    p2 = MemoryProposal(
        task_id="task-1",
        agent_id="agent-b",
        source="worker",
        step_index=2,
        title="Transaction Isolation",
        raw_value="Lock contention orders isolation level causes deadlock",
        proposal_id="p2",
    )
    adapter.submit(p2)

    # p2 should link to p1 due to token overlap ("lock", "contention", "orders")
    n1 = adapter.read("p1", "agent-a", "task-1")
    n2 = adapter.read("p2", "agent-b", "task-1")
    assert "p1" in n2.links
    assert "p2" in n1.links

    # Box expansion in retrieval
    keys = adapter.visible_keys("agent-c", "task-1", query="isolation", top_k=5)
    key_ids = [k.memory_id for k in keys]
    # Primary hit is p2, but p1 should be expanded into visible keys
    assert "p2" in key_ids
    assert "p1" in key_ids


def test_memoryos_adapter_paging_and_heat():
    adapter = MemoryOSAdapter(stm_capacity=2, mtm_capacity=2)
    for i in range(4):
        p = MemoryProposal(
            task_id="task-1",
            agent_id="agent-a",
            source="worker",
            step_index=i,
            title=f"Event {i}",
            raw_value=f"Log trace line {i}",
            proposal_id=f"p{i}",
        )
        adapter.submit(p)

    # First 2 items should have moved to MTM
    assert adapter._items["p0"].tier in ("MTM", "LTM")
    assert adapter._items["p1"].tier in ("MTM", "LTM")
    assert adapter._items["p2"].tier == "STM"
    assert adapter._items["p3"].tier == "STM"

    # Heat increments on read
    h_before = adapter._items["p2"].heat
    adapter.read("p2", "agent-b", "task-1")
    assert adapter._items["p2"].heat > h_before


def test_make_memory_runtime_instantiates_classical(tmp_path):
    trace_file = tmp_path / "trace.jsonl"
    m0 = make_memory_runtime("mem0", trace_file)
    assert isinstance(m0, Mem0Adapter)

    am = make_memory_runtime("amem", trace_file)
    assert isinstance(am, AMemAdapter)

    mos = make_memory_runtime("memoryos", trace_file)
    assert isinstance(mos, MemoryOSAdapter)
