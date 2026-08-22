import json
import os

from marble.experiments.baselines import BASELINES
from marble.experiments.coding_rollout import extract_code, run_baselines, run_episode


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def act(self, system: str, user: str) -> str:
        self.calls += 1
        return "```python\nprint('step %d')\n```" % self.calls


def _task():
    return {"t1": "Write solution.py printing hello."}


def test_extract_code_prefers_last_fence():
    assert extract_code("a\n```python\nx=1\n```\n```python\ny=2\n```") == "y=2"
    assert extract_code("no fence") == "no fence"


def test_run_episode_global_shares_and_reads(tmp_path):
    stats = run_episode("t1", _task()["t1"], FakeLLM(), None, str(tmp_path / "ws"))
    assert stats["proposals_stored"] == 0
    assert os.path.exists(stats["solution_path"])


def test_run_baselines_offline_smoke(tmp_path):
    summary = run_baselines(list(BASELINES), str(tmp_path), FakeLLM(), tasks=_task())
    for baseline in BASELINES:
        results = summary[baseline]
        assert len(results) == 1
        assert os.path.exists(results[0]["solution_path"])
    with open(os.path.join(str(tmp_path), "summary.json")) as fh:
        persisted = json.load(fh)
    assert set(persisted.keys()) == set(BASELINES)
