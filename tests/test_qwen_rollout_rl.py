import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from marble.experiments import qwen_rollout_rl


def _complete_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"adapter")


def _events(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (path / "training_events.jsonl").read_text().splitlines()
    ]


def test_train_fresh_rollouts_skips_complete_round_and_resumes(tmp_path, monkeypatch):
    initial = tmp_path / "initial"
    _complete_checkpoint(initial)
    root = tmp_path / "training"
    round_zero = root / "round-000"
    round_zero.mkdir(parents=True)
    (round_zero / "rollout_manifest.json").write_text(
        json.dumps({"round": 0, "rollouts": []}), encoding="utf-8"
    )
    _complete_checkpoint(root / "checkpoint-001")

    tasks = [SimpleNamespace(task_id=1, benchmark="coding")]
    monkeypatch.setattr(qwen_rollout_rl, "load_tasks", lambda *args, **kwargs: tasks)
    monkeypatch.setattr(qwen_rollout_rl, "_apply_split", lambda value, split: value)
    collected = []

    def collect(*args, **kwargs):
        collected.append((args[1], args[2]))
        return ["trace-a", "trace-b"], [1.0, 0.5], []

    def train(*args, **kwargs):
        _complete_checkpoint(Path(args[1]))

    monkeypatch.setattr(qwen_rollout_rl, "collect_rollouts", collect)
    monkeypatch.setattr(qwen_rollout_rl, "train_qwen_rl", train)

    result = qwen_rollout_rl.train_fresh_rollouts(
        "coding",
        str(initial),
        str(root),
        rounds=2,
    )

    assert [item[1].name for item in collected] == ["round-001"]
    assert result["final_checkpoint"] == str((root / "checkpoint-002").resolve())
    assert result["status"] == "completed"
    assert [event["event"] for event in _events(root)] == [
        "start",
        "round_skip",
        "round_start",
        "round_end",
        "end",
    ]


def test_resume_uses_checkpoint_from_training_manifest(tmp_path, monkeypatch):
    initial = tmp_path / "initial"
    _complete_checkpoint(initial)
    root = tmp_path / "training"
    round_zero = root / "round-000"
    round_zero.mkdir(parents=True)
    (round_zero / "rollout_manifest.json").write_text("{}", encoding="utf-8")
    prior_checkpoint = tmp_path / "prior-adapter"
    _complete_checkpoint(prior_checkpoint)
    (root / "training_manifest.json").write_text(
        json.dumps(
            {
                "rounds_detail": [
                    {
                        "round": 0,
                        "output_checkpoint": str(prior_checkpoint),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    tasks = [SimpleNamespace(task_id=1, benchmark="coding")]
    monkeypatch.setattr(qwen_rollout_rl, "load_tasks", lambda *args, **kwargs: tasks)
    monkeypatch.setattr(qwen_rollout_rl, "_apply_split", lambda value, split: value)
    observed = {}

    def collect(*args, **kwargs):
        observed["checkpoint"] = args[1]
        return ["trace-a", "trace-b"], [1.0, 0.5], []

    def train(*args, **kwargs):
        _complete_checkpoint(Path(args[1]))

    monkeypatch.setattr(qwen_rollout_rl, "collect_rollouts", collect)
    monkeypatch.setattr(qwen_rollout_rl, "train_qwen_rl", train)

    qwen_rollout_rl.train_fresh_rollouts(
        "coding",
        str(initial),
        str(root),
        rounds=2,
    )

    assert observed["checkpoint"] == str(prior_checkpoint.resolve())


def test_train_fresh_rollouts_records_error_and_keeps_manifest_valid(tmp_path, monkeypatch):
    initial = tmp_path / "initial"
    _complete_checkpoint(initial)
    root = tmp_path / "training"
    tasks = [SimpleNamespace(task_id=1, benchmark="coding")]
    monkeypatch.setattr(qwen_rollout_rl, "load_tasks", lambda *args, **kwargs: tasks)
    monkeypatch.setattr(qwen_rollout_rl, "_apply_split", lambda value, split: value)
    monkeypatch.setattr(
        qwen_rollout_rl,
        "collect_rollouts",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        qwen_rollout_rl.train_fresh_rollouts(
            "coding",
            str(initial),
            str(root),
        )

    manifest = json.loads((root / "training_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert [event["event"] for event in _events(root)] == [
        "start",
        "round_start",
        "error",
        "end",
    ]
