import hashlib
import json
import os

from marble.controllers import LocalPolicyController
from marble.experiments.train_controller import (
    accuracy,
    discover_sft_traces,
    load_samples,
    train,
)


def _trace(tmp_path, name, rows):
    path = str(tmp_path / name)
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _decision(pid, title, value, visibility):
    return {
        "event": "memory_decision",
        "memory_id": pid,
        "proposal": {"proposal_id": pid, "task_id": "t", "agent_id": "a",
                     "source": "worker", "title": title, "raw_value": value,
                     "step_index": 1},
        "target": {"exists": visibility != "absent", "visibility": visibility,
                   "supersedes": None},
    }


def test_load_samples_filters_and_parses(tmp_path):
    good = str(tmp_path / "good.jsonl")
    with open(good, "w") as fh:
        fh.write(json.dumps(_decision("m1", "shared result", "x=1", "global")) + "\n")
        fh.write(json.dumps({"event": "memory_read"}) + "\n")
        fh.write(json.dumps({"event": "memory_decision"}) + "\n")  # no memory_id
    samples = load_samples([good])
    assert len(samples) == 1
    assert samples[0]["label"] == "global"


def test_load_samples_preserves_absent_without_memory_id(tmp_path):
    trace_file = str(tmp_path / "absent_trace.jsonl")
    with open(trace_file, "w") as fh:
        # Realistic absent event as emitted by GovernedMemory.submit when memory_id is None
        absent_ev = {
            "event": "memory_decision",
            "memory_id": None,
            "proposal": {
                "proposal_id": "p1", "task_id": "t", "agent_id": "a",
                "source": "worker", "title": "scratch note", "raw_value": "x",
                "step_index": 1,
            },
            "target": {"exists": False, "visibility": "absent", "supersedes": None},
        }
        fh.write(json.dumps(absent_ev) + "\n")
    samples = load_samples([trace_file])
    assert len(samples) == 1
    assert samples[0]["label"] == "absent"


def test_train_improves_accuracy_on_separable_data(tmp_path):
    rows = []
    for i in range(10):
        rows.append(_decision(f"g{i}", f"shared plan {i}", "v", "global"))
        rows.append(_decision(f"p{i}", f"private note {i}", "v", "private"))
        rows.append(_decision(f"a{i}", f"note {i}", "", "absent"))
    trace = _trace(tmp_path, "trace.jsonl", rows)
    out = str(tmp_path / "policy.json")
    stats = train([trace], out, epochs=30)
    assert stats["accuracy_after"] > stats["accuracy_before"]
    assert os.path.exists(out)
    policy = LocalPolicyController.load(out)
    samples = load_samples([trace])
    assert accuracy(policy, samples) == 1.0


def _episode(tmp_path, method, task_id, score_status="available", benchmark="coding"):
    task_dir = tmp_path / method / benchmark / str(task_id)
    task_dir.mkdir(parents=True)
    trace = task_dir / "memory_trace.jsonl"
    trace.write_text(json.dumps(_decision(f"m{task_id}", "shared result", "x=1", "global")) + "\n")
    (task_dir / "summary.json").write_text(json.dumps({
        "method": method,
        "benchmark": benchmark,
        "task_id": task_id,
        "score_status": score_status,
    }))
    return str(trace)


def test_discover_sft_traces_uses_train_split_and_canonical_baselines(tmp_path):
    train_task = next(
        task_id for task_id in range(100)
        if int(hashlib.sha256(str(task_id).encode()).hexdigest(), 16) % 10 < 8
    )
    other_train_task = next(
        task_id for task_id in range(train_task + 1, 100)
        if int(hashlib.sha256(str(task_id).encode()).hexdigest(), 16) % 10 < 8
    )
    test_task = next(
        task_id for task_id in range(100)
        if int(hashlib.sha256(str(task_id).encode()).hexdigest(), 16) % 10 >= 8
    )
    included = _episode(tmp_path, "qwen_sft", train_task)
    _episode(tmp_path, "ours_sft", test_task)
    _episode(tmp_path, "ours_sft", other_train_task, score_status="unavailable")
    _episode(tmp_path, "heuristic", train_task + 200)

    traces = discover_sft_traces([str(tmp_path)], baseline="ours_sft", split="train")

    assert traces == [os.path.abspath(included)]


def test_discover_sft_traces_keeps_legacy_direct_trace_paths(tmp_path):
    trace = _trace(tmp_path, "legacy.jsonl", [_decision("m1", "shared result", "x=1", "global")])

    assert discover_sft_traces([trace]) == [os.path.abspath(trace)]


def test_discover_sft_traces_uses_frozen_manifest_split(tmp_path):
    train_trace = _episode(tmp_path, "ours_sft", 1, benchmark="research")
    test_trace = _episode(tmp_path, "ours_sft", 2, benchmark="research")
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps({
        "splits": {
            "train": [{"benchmark": "research", "task_id": 1}],
            "test": [{"benchmark": "research", "task_id": 2}],
        }
    }), encoding="utf-8")

    assert discover_sft_traces([str(tmp_path)], manifest=str(manifest), split="train") == [
        os.path.abspath(train_trace)
    ]
    assert discover_sft_traces([str(tmp_path)], manifest=str(manifest), split="test") == [
        os.path.abspath(test_trace)
    ]


def test_manifest_keeps_direct_trace_compatibility(tmp_path):
    trace = _trace(tmp_path, "legacy.jsonl", [_decision("m1", "shared result", "x=1", "global")])
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps({"splits": {"train": [], "test": []}}), encoding="utf-8")

    assert discover_sft_traces([trace], manifest=str(manifest), split="train") == [
        os.path.abspath(trace)
    ]


def test_load_samples_and_train_multi_agent_recipients(tmp_path):
    rows = []
    # Mix of global, absent, and multi-agent targeted decisions
    for i in range(10):
        rows.append({
            "event": "memory_decision",
            "memory_id": f"m_pair_{i}",
            "proposal": {"proposal_id": f"p{i}", "task_id": "t", "agent_id": "agent_1",
                         "source": "worker", "title": f"pair sync {i}", "raw_value": "state",
                         "step_index": 1},
            "target": {"exists": True, "visibility": ["agent_1", "agent_2"], "supersedes": None},
        })
        rows.append({
            "event": "memory_decision",
            "memory_id": f"m_single_{i}",
            "proposal": {"proposal_id": f"s{i}", "task_id": "t", "agent_id": "agent_2",
                         "source": "worker", "title": f"internal note {i}", "raw_value": "draft",
                         "step_index": 1},
            "target": {"exists": True, "visibility": "targeted", "target_recipients": ["agent_2"], "supersedes": None},
        })
        rows.append(_decision(f"g{i}", f"shared plan {i}", "all agree", "global"))
        rows.append(_decision(f"a{i}", f"empty {i}", "", "absent"))

    trace = _trace(tmp_path, "multi_trace.jsonl", rows)
    samples = load_samples([trace])
    assert len(samples) == 40
    labels = {s["label"] for s in samples}
    assert '["agent_1", "agent_2"]' in labels
    assert '["agent_2"]' in labels
    assert "global" in labels
    assert "absent" in labels

    out = str(tmp_path / "multi_policy.json")
    stats = train([trace], out, epochs=30)
    assert stats["accuracy_after"] >= 0.9
    assert os.path.exists(out)


def test_train_controller_cli_help():
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "-m", "marble.experiments.train_controller", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "linear_sft" in res.stdout
    assert "linear_rl" in res.stdout
    assert "qwen_sft" in res.stdout
    assert "qwen_rl" in res.stdout
    assert "GRPO" in res.stdout
    assert "sft-qwen" not in res.stdout
    assert "rl-qwen" not in res.stdout


def test_accuracy_evaluates_correctly_and_queries_policy_once_per_sample():
    class CountingPolicy:
        def __init__(self):
            self.call_count = 0

        def scores(self, feat, proposal=None):
            self.call_count += 1
            return {"global": 0.8, "targeted": 0.2, "absent": 0.1}

    policy = CountingPolicy()
    samples = [
        {"feat": {"f": 1}, "label": "global", "proposal": {}},
        {"feat": {"f": 2}, "label": "absent", "proposal": {}},
    ]
    acc = accuracy(policy, samples)
    assert acc == 0.5
    assert policy.call_count == 2


