import pytest

from marble.benchmarks import BENCHMARKS, BenchmarkTask, load_tasks


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


def test_load_real_coding_benchmark():
    tasks = load_tasks("coding", limit=3)
    assert len(tasks) == 3
    t = tasks[0]
    assert isinstance(t, BenchmarkTask)
    assert t.benchmark == "coding"
    assert isinstance(t.task_id, int) and isinstance(t.task, str) and t.task
    assert t.agents and t.agents[0]["agent_id"]


def test_minecraft_missing_optional_keys_default():
    # minecraft records ship no task_id/scenario; memory is SharedMemory
    tasks = load_tasks("minecraft", limit=1)
    t = tasks[0]
    assert t.scenario == "" and isinstance(t.task_id, int)
    assert t.memory == {"type": "SharedMemory"}


def test_limit_start_and_task_ids(tmp_path):
    recs = [_rec(i) for i in range(5)]
    p = _write(tmp_path, recs)
    assert [t.task_id for t in load_tasks("coding", path=p)] == [0, 1, 2, 3, 4]
    assert [t.task_id for t in load_tasks("coding", path=p, start=2, limit=2)] == [2, 3]
    assert [t.task_id for t in load_tasks("coding", path=p, task_ids=[4, 1])] == [1, 4]


def test_unknown_benchmark_raises():
    with pytest.raises(ValueError):
        load_tasks("nope")


def test_all_benchmarks_present():
    assert set(BENCHMARKS) >= {"coding", "research", "database", "bargaining", "minecraft"}


def test_frozen_manifest_has_expected_splits():
    from marble.experiments.task_manifest import load_manifest_tasks

    path = __import__("pathlib").Path(__file__).parents[1] / "configs/experiments/multiagentbench_frozen.json"
    train = load_manifest_tasks(path, split="train")
    test = load_manifest_tasks(path, split="test")
    assert len(train) == 6
    assert len(test) == 18
    assert {task.benchmark for task in train + test} == {"database", "research"}
    assert {len(task.agents) for task in train + test} <= {3, 4, 5}


def test_stratified_frozen_manifest_has_expected_splits():
    from marble.experiments.task_manifest import load_manifest_tasks

    path = __import__("pathlib").Path(__file__).parents[1] / "configs/experiments/multiagentbench_stratified_frozen.json"
    train_std = load_manifest_tasks(path, split="train_standard")
    train_hard = load_manifest_tasks(path, split="train_hard")
    test_std = load_manifest_tasks(path, split="test_standard")
    test_hard = load_manifest_tasks(path, split="test_hard")

    assert len(train_std) == 6
    assert len(train_hard) == 6
    assert len(test_std) == 8
    assert len(test_hard) == 8

    # Strict zero-leakage check
    train_keys = {(t.benchmark, t.task_id) for t in train_std + train_hard}
    test_keys = {(t.benchmark, t.task_id) for t in test_std + test_hard}
    assert train_keys.isdisjoint(test_keys)

