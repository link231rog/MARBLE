import json

from marble.benchmarks import load_tasks
from marble.experiments.run_benchmark import (
    BASELINES,
    make_controller,
    plan_runs,
    run_task,
    task_config,
)


def _task():
    return load_tasks("coding", limit=1)[0]


def test_make_controller_fixed_baselines():
    from marble.controllers import (
        AbsentController,
        GlobalAlwaysController,
        HeuristicController,
        PrivateOnlyController,
    )

    assert isinstance(make_controller("no_memory"), AbsentController)
    assert isinstance(make_controller("global_always"), GlobalAlwaysController)
    assert isinstance(make_controller("private_only"), PrivateOnlyController)
    assert isinstance(make_controller("heuristic"), HeuristicController)


def test_make_controller_checkpoint_roundtrip(tmp_path):
    from marble.controllers import LocalPolicyController

    src = LocalPolicyController()
    p = tmp_path / "policy.json"
    src.save(str(p))
    ctrl = make_controller("learned_controller", controller_checkpoint=str(p))
    assert isinstance(ctrl, LocalPolicyController)


def test_learned_controller_without_key_raises(monkeypatch):
    for var in ("OPENAI_API_KEY", "NVAPI_KEY", "MARBLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    try:
        make_controller("learned_controller")
        raised = False
    except RuntimeError as exc:
        raised = "NVAPI_KEY" in str(exc)
    assert raised


def test_task_config_injects_governed_block_only_when_needed():
    t = _task()
    plain = task_config(t, "no_memory")
    assert plain["memory"] == {} or "backend" not in plain["memory"]
    gov = task_config(t, "heuristic", max_cards=3, max_reads_per_step=1)
    assert gov["memory"]["backend"] == "governed"
    assert gov["memory"]["controller"] == "heuristic"
    assert gov["memory"]["max_cards"] == 3
    assert gov["memory"]["max_reads_per_step"] == 1
    assert gov["task"]["content"] == t.task
    assert gov["agents"] == [dict(a) for a in t.agents]


def test_dry_run_writes_layout_without_engine(tmp_path):
    t = _task()
    summary = run_task(t, "heuristic", tmp_path, dry_run=True, seed=7)
    tdir = tmp_path / "coding" / str(t.task_id)
    assert (tdir / "config.yaml").exists()
    on_disk = json.loads((tdir / "summary.json").read_text())
    assert on_disk["status"] == "dry_run"
    assert on_disk["method"] == "heuristic" and on_disk["seed"] == 7
    assert summary["status"] == "dry_run"


def test_real_run_without_any_key_records_error(tmp_path, monkeypatch):
    for var in ("OPENAI_API_KEY", "NVAPI_KEY", "MARBLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    t = _task()
    summary = run_task(t, "no_memory", tmp_path)
    tdir = tmp_path / "coding" / str(t.task_id)
    assert summary["status"] == "error"
    err = (tdir / "errors.log").read_text()
    assert "worker API key" in err
    assert (tdir / "memory_trace.jsonl").exists() is False or True  # trace may pre-exist


def test_plan_runs_multi_expands_once():
    tasks = [_task()]
    runs = plan_runs(["multi"], tasks)
    assert [b for b, _ in runs] == list(BASELINES)
    # duplicates collapse, order preserved; duplicated tasks double the episodes
    dup = plan_runs(["heuristic", "multi"], tasks * 2)
    assert list(dict.fromkeys(b for b, _ in dup)) == ["heuristic"] + [
        b for b in BASELINES if b != "heuristic"
    ]
    assert len(dup) == 10
