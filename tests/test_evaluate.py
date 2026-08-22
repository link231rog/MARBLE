import json
import os

from marble.experiments.baselines import BASELINES
from marble.experiments.coding_rollout import run_baselines
from marble.experiments.evaluate import evaluate_baseline, evaluate_run, format_report


class FakeLLM:
    def act(self, system: str, user: str) -> str:
        return "```python\nprint('hi')\n```"


def test_evaluate_run_offline(tmp_path):
    run_dir = str(tmp_path)
    names = [str(b) for b in BASELINES][:2]
    run_baselines(names, run_dir, FakeLLM(),
                  tasks={"t1": "Write solution.py."})
    report = evaluate_run(run_dir)
    assert set(report.keys()) == {"no_memory", "global_always"}
    for metrics in report.values():
        assert metrics["episodes"] == 1
        assert metrics["compile_rate"] == 1.0
    # global_always stores everything as global; no_memory stores nothing
    assert report["global_always"]["decisions"]["global"] == 2
    assert sum(report["no_memory"]["decisions"].values()) == 0
    text = format_report(report)
    assert "global_always" in text and "compile_rate" in text


def test_evaluate_baseline_empty_trace(tmp_path):
    entry = [{"proposals_stored": 0, "visible_key_events": 0, "reads": [],
              "compile_ok": False}]
    m = evaluate_baseline(entry, str(tmp_path / "missing.jsonl"))
    assert m["episodes"] == 1 and m["compile_rate"] == 0.0
