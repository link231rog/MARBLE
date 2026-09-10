import pathlib
import pytest
from marble.benchmarks import BENCHMARKS, BenchmarkTask, load_tasks
from marble.experiments.task_manifest import load_manifest_tasks


def _write(tmp_path, records):
    p = tmp_path / "fake_main.jsonl"
    p.write_text("\n".join(__import__("json").dumps(r) for r in records), encoding="utf-8")
    return p


def _rec(task_id=0):
    return {
        "scenario": "coding",
        "task_id": task_id,
        "task": {"content": f"do {task_id}"},
        "agents": [{"type": "CodingAgent", "agent_id": "agent1"}],
        "relationships": [["agent1", "agent2", "collaborate"]],
        "environment": {"max_iterations": 3},
    }


def test_load_tasks_and_benchmarks(tmp_path):
    # Benchmark sets
    assert set(BENCHMARKS) >= {"coding", "research", "database", "bargaining", "minecraft"}
    with pytest.raises(ValueError):
        load_tasks("nope")

    # Real coding benchmark
    tasks = load_tasks("coding", limit=3)
    assert len(tasks) == 3
    t = tasks[0]
    assert isinstance(t, BenchmarkTask)
    assert t.benchmark == "coding"
    assert isinstance(t.task_id, int) and isinstance(t.task, str) and t.task
    assert t.agents and t.agents[0]["agent_id"]

    # Minecraft default keys
    tasks_mc = load_tasks("minecraft", limit=1)
    assert tasks_mc[0].scenario == "" and isinstance(tasks_mc[0].task_id, int)
    assert tasks_mc[0].memory == {"type": "SharedMemory"}

    # Start, limit, task_ids
    recs = [_rec(i) for i in range(5)]
    p = _write(tmp_path, recs)
    assert [tk.task_id for tk in load_tasks("coding", path=p)] == [0, 1, 2, 3, 4]
    assert [tk.task_id for tk in load_tasks("coding", path=p, start=2, limit=2)] == [2, 3]
    assert [tk.task_id for tk in load_tasks("coding", path=p, task_ids=[4, 1])] == [1, 4]


def test_task_manifest_splits():
    base_dir = pathlib.Path(__file__).parents[1] / "configs/experiments"

    # 1. multiagentbench_frozen.json
    path1 = base_dir / "multiagentbench_frozen.json"
    train = load_manifest_tasks(path1, split="train")
    test = load_manifest_tasks(path1, split="test")
    assert len(train) == 6
    assert len(test) == 18

    # 2. multiagentbench_stratified_frozen.json
    path2 = base_dir / "multiagentbench_stratified_frozen.json"
    train_std = load_manifest_tasks(path2, split="train_standard")
    train_hard = load_manifest_tasks(path2, split="train_hard")
    test_std = load_manifest_tasks(path2, split="test_standard")
    test_hard = load_manifest_tasks(path2, split="test_hard")
    assert len(train_std) == 9 and len(train_hard) == 12
    assert len(test_std) == 12 and len(test_hard) == 24
    train_keys = {(t.benchmark, t.task_id) for t in train_std + train_hard}
    test_keys = {(t.benchmark, t.task_id) for t in test_std + test_hard}
    assert train_keys.isdisjoint(test_keys)

    # 3. multiagentbench_hard_frozen.json
    path3 = base_dir / "multiagentbench_hard_frozen.json"
    assert len(load_manifest_tasks(path3, split="train_hard")) == 12
    assert len(load_manifest_tasks(path3, split="test_hard")) == 24
