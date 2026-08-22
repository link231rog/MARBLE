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
    tdir = tmp_path / "heuristic" / "coding" / str(t.task_id)
    assert (tdir / "config.yaml").exists()
    on_disk = json.loads((tdir / "summary.json").read_text())
    assert on_disk["status"] == "dry_run"
    assert on_disk["method"] == "heuristic" and on_disk["seed"] == 7
    assert summary["status"] == "dry_run"


def test_multi_run_keeps_every_method_on_disk(tmp_path):
    tasks = [_task()]
    for baseline, task in plan_runs(["multi"], tasks):
        run_task(task, baseline, tmp_path, dry_run=True)
    found = sorted(
        (json.loads(p.read_text())["method"], p.parent)
        for p in tmp_path.rglob("summary.json")
    )
    assert [m for m, _ in found] == sorted(BASELINES)  # no overwrites
    assert len({d for _, d in found}) == len(BASELINES)


def test_real_run_without_any_key_records_error(tmp_path, monkeypatch):
    for var in ("OPENAI_API_KEY", "NVAPI_KEY", "MARBLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    t = _task()
    summary = run_task(t, "no_memory", tmp_path)
    tdir = tmp_path / "no_memory" / "coding" / str(t.task_id)
    assert summary["status"] == "error"
    err = (tdir / "errors.log").read_text()
    assert "worker API key" in err
    assert not (tdir / "memory_trace.jsonl").exists()  # no trace before real episode


def test_task_config_uses_coordinate_mode_key_and_wires_llm():
    # regression: Config reads coordinate_mode; coordination_mode would crash
    t = _task()
    cfg = task_config(t, "heuristic", llm="gpt-4o-mini")
    assert cfg["coordinate_mode"] == "graph"
    assert "coordination_mode" not in cfg
    assert cfg["llm"] == "gpt-4o-mini"


def test_apply_split_deterministic():
    from marble.experiments.run_benchmark import _apply_split

    tasks = [_task()] * 3 + load_tasks("coding", limit=40)[1:4]
    train = _apply_split(tasks, "train")
    test = _apply_split(tasks, "test")
    assert {t.task_id for t in train} | {t.task_id for t in test} == {
        t.task_id for t in tasks
    }
    # stable: same split returns same membership
    assert {t.task_id for t in _apply_split(tasks, "train")} == {
        t.task_id for t in train
    }
    assert _apply_split(tasks, "all") == tasks


def test_ablation_wired_into_task_config():
    from marble.experiments.run_benchmark import apply_to_task_config, parse_ablation

    t = _task()
    cfg = task_config(t, "learned_controller", ablation="retrieval:none")
    # parse_ablation accepted; retrieval ablation marked in memory block
    assert cfg["memory"]["ablation"] == "retrieval:none"
    assert cfg["memory"]["max_cards"] == 0  # retrieval:none zeroes visible cards
