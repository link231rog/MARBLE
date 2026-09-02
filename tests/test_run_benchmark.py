import json
import re

from marble.benchmarks import load_tasks
from marble.controllers import LocalPolicyController
from marble.controllers import qwen_lora
from marble.experiments.run_benchmark import (
    BASELINES,
    make_controller,
    make_governed,
    plan_runs,
    run_task,
    task_config,
)
from marble.memory.schema import MemoryProposal


def _task():
    return load_tasks("coding", limit=1)[0]


def test_make_controller_checkpoint_roundtrip(tmp_path):
    from marble.controllers import LocalPolicyController

    src = LocalPolicyController()
    p = tmp_path / "policy.json"
    src.save(str(p))
    ctrl = make_controller("learned_controller", controller_checkpoint=str(p))
    assert isinstance(ctrl, LocalPolicyController)


def test_make_governed_drops_worker_runtime_argument(tmp_path):
    runtime = make_governed(
        "heuristic", tmp_path / "trace.jsonl", worker_model="deepseek-v4-flash"
    )
    assert runtime.controller is not None


def test_learned_controller_unified_local_policy_without_key(monkeypatch):
    for var in ("OPENAI_API_KEY", "NVAPI_KEY", "MARBLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # review #3: learned_controller is unified to LocalPolicyController (no API
    # key needed); untrained -> absent. No RuntimeError expected here.
    ctrl = make_controller("learned_controller")
    assert isinstance(ctrl, LocalPolicyController)


def test_qwen_local_mode_loads_adapter_without_overloading_learned_controller(monkeypatch):
    seen = {}

    def fake_local(base_model, lora_dir, temperature=0.0):
        seen["base_model"] = base_model
        seen["lora_dir"] = lora_dir
        seen["temperature"] = temperature
        return lambda prompt: '{"visibility":"private","supersedes":null}'

    monkeypatch.setattr(qwen_lora, "local_generate_fn", fake_local)
    ctrl = make_controller(
        "qwen_sft",
        controller_checkpoint="/adapters/sft",
        qwen_base_model="/models/qwen",
    )
    assert seen == {
        "base_model": "/models/qwen",
        "lora_dir": "/adapters/sft",
        "temperature": 0.0,
    }
    assert ctrl.decide(
        MemoryProposal(
            proposal_id="p", task_id="t", agent_id="a", source="worker",
            title="x", raw_value="y", step_index=0,
        ),
        [],
    ).visibility == "private"


def test_qwen_sft_ablation_passes_drop_fields(monkeypatch):
    """Input/schema ablations must reach make_qwen_lora_controller as drop_fields."""
    seen = {}
    monkeypatch.delenv("MARBLE_QWEN_API_BASE", raising=False)

    def fake_local(base_model, lora_dir, temperature=0.0):
        return lambda prompt: '{"visibility":"private","supersedes":null}'

    def fake_make(gen, drop_fields=(), **kw):
        seen["drop_fields"] = drop_fields
        from marble.controllers.json_controller import JsonController
        return JsonController(gen, drop_fields=drop_fields)

    monkeypatch.setattr(qwen_lora, "local_generate_fn", fake_local)
    from marble.controllers import qwen_lora as _ql
    monkeypatch.setattr(_ql, "make_qwen_lora_controller", fake_make)

    ctrl = make_controller(
        "qwen_sft",
        controller_checkpoint="/adapters/sft",
        qwen_base_model="/models/qwen",
        ablation="input:no_agent_tag",
    )
    assert seen["drop_fields"] == ("agent_tag",)
    # Verify the controller actually removes the agent tag from its prompt.
    prompt = ctrl.build_prompt(
        MemoryProposal(
            proposal_id="p", task_id="t", agent_id="a", source="worker",
            title="x", raw_value="y", step_index=0,
        ),
        [],
    )
    assert "agent_reference" not in prompt


def test_qwen_rl_schema_field_ablation_passes_drop_fields(monkeypatch):
    """schema_field ablation passes the correct drop_fields to Qwen controller."""
    seen = {}

    def fake_api(api_base, api_key, model):
        return lambda prompt: '{"visibility":"global","supersedes":null}'

    def fake_make(gen, drop_fields=(), **kw):
        seen["drop_fields"] = drop_fields
        from marble.controllers.json_controller import JsonController
        return JsonController(gen, drop_fields=drop_fields)

    monkeypatch.setattr(qwen_lora, "api_generate_fn", fake_api)
    from marble.controllers import qwen_lora as _ql
    monkeypatch.setattr(_ql, "make_qwen_lora_controller", fake_make)

    ctrl = make_controller(
        "qwen_rl",
        qwen_api_base="https://example/v1",
        qwen_api_key="k",
        ablation="schema_field:title",
    )
    assert seen["drop_fields"] == ("title",)


def test_non_qwen_ablation_does_not_double_apply(monkeypatch):
    """Non-Qwen baselines must not be affected by input/schema ablation kwargs."""
    ctrl = make_controller("heuristic", ablation="input:no_agent_tag")
    # heuristic is HeuristicController, not JsonController — no drop_fields
    assert not hasattr(ctrl, "drop_fields")


def test_qwen_endpoint_mode_uses_explicit_served_model(monkeypatch):
    seen = {}

    def fake_api(api_base, api_key, model):
        seen.update(api_base=api_base, api_key=api_key, model=model)
        return lambda prompt: '{"visibility":"global","supersedes":null}'

    monkeypatch.setattr(qwen_lora, "api_generate_fn", fake_api)
    ctrl = make_controller(
        "qwen_rl",
        qwen_api_base="https://controller.example/v1",
        qwen_api_key="test-key",
        qwen_api_model="qwen-controller-rl",
    )
    assert seen == {
        "api_base": "https://controller.example/v1",
        "api_key": "test-key",
        "model": "qwen-controller-rl",
    }
    assert ctrl is not None


def test_memory_r1_runtime_uses_global_crud_adapter(tmp_path):
    from marble.memory import MemoryR1Adapter
    from marble.experiments.run_benchmark import make_memory_runtime

    runtime = make_memory_runtime("memory_r1_style", tmp_path / "trace.jsonl")
    assert isinstance(runtime, MemoryR1Adapter)
    assert runtime.controller is None


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
    # worker-precedence: task_config forces llm on every agent; original fields kept
    assert len(gov["agents"]) == len(t.agents)
    for orig, injected in zip(t.agents, gov["agents"]):
        assert {k: v for k, v in injected.items() if k != "llm"} == dict(orig)
        assert injected["llm"]


def test_single_agent_task_config_keeps_only_one_agent():
    t = _task()
    cfg = task_config(t, "single_agent")

    assert len(cfg["agents"]) == 1
    chosen_id = cfg["agents"][0]["agent_id"]
    assert all(chosen_id in relation for relation in cfg["relationships"])
    assert "backend" not in cfg["memory"]


def test_task_config_overrides_empty_env_type():
    # JSONL ships environment.type="" which would make Engine raise
    # "Unsupported environment type"; task_config must fill the benchmark default.
    t = _task()
    assert not str(t.environment.get("type", "")).strip()
    cfg = task_config(t, "heuristic")
    assert cfg["environment"]["type"] == "Coding"
    assert cfg["environment"]["max_iterations"] == 10


def test_dry_run_writes_layout_without_engine(tmp_path):
    t = _task()
    summary = run_task(
        t,
        "heuristic",
        tmp_path,
        dry_run=True,
        seed=7,
        max_cards=4,
        max_reads_per_step=1,
        lambda_=0.1,
        beta=0.5,
    )
    tdir = tmp_path / "heuristic" / "coding" / str(t.task_id)
    assert (tdir / "config.yaml").exists()
    on_disk = json.loads((tdir / "summary.json").read_text())
    assert on_disk["status"] == "dry_run"
    assert on_disk["method"] == "heuristic" and on_disk["seed"] == 7
    assert on_disk["agent_count"] == len(t.agents)
    assert on_disk["ablation"] is None and on_disk["manifest"] is None
    assert on_disk["setting"].startswith("method=heuristic|ablation=none|")
    assert summary["status"] == "dry_run"
    # spec §11.2/§11.3: model names recorded, all non-empty
    assert on_disk["worker_model"] and on_disk["controller_model"] == "heuristic"
    assert on_disk["evaluator_model"]
    assert on_disk["retrieval"] == {
        "name": "key_first",
        "max_cards": 4,
        "max_reads_per_step": 1,
    }
    assert on_disk["reward_config"] == {"lambda": 0.1, "beta": 0.5}


def test_explicit_out_is_used_as_run_root_and_writes_events(tmp_path):
    t = _task()
    run_task(t, "heuristic", tmp_path, dry_run=True)

    assert (tmp_path / "heuristic" / "coding" / str(t.task_id) / "summary.json").exists()
    events = [
        json.loads(line)
        for line in (tmp_path / "run_events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["task_start", "task_end"]
    assert all(event["task_id"] == t.task_id for event in events)


def test_completed_task_is_skipped_and_logged(tmp_path, monkeypatch):
    t = _task()
    tdir = tmp_path / "heuristic" / "coding" / str(t.task_id)
    tdir.mkdir(parents=True)
    completed = {"status": "ok", "method": "heuristic", "task_id": t.task_id}
    (tdir / "summary.json").write_text(json.dumps(completed))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("_run_real_episode must not run for completed tasks")

    monkeypatch.setattr("marble.experiments.run_benchmark._run_real_episode", fail_if_called)
    assert run_task(t, "heuristic", tmp_path) == completed

    events = [
        json.loads(line)
        for line in (tmp_path / "run_events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["task_start", "task_skip"]
    assert events[-1]["status"] == "ok"


def test_failed_task_is_rerun_and_end_event_is_appended(tmp_path, monkeypatch):
    t = _task()
    tdir = tmp_path / "heuristic" / "coding" / str(t.task_id)
    tdir.mkdir(parents=True)
    (tdir / "summary.json").write_text(json.dumps({"status": "error"}))
    monkeypatch.setattr(
        "marble.experiments.run_benchmark._run_real_episode",
        lambda *args, **kwargs: {"score_status": "available", "task_score": 1.0},
    )

    summary = run_task(t, "heuristic", tmp_path)
    assert summary["status"] == "ok"
    events = [
        json.loads(line)
        for line in (tmp_path / "run_events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["task_start", "task_end"]
    assert events[-1]["status"] == "ok"


def test_default_runs_get_collision_free_timestamped_roots(tmp_path, monkeypatch):
    from marble.experiments import run_benchmark

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_benchmark, "load_tasks", lambda *args, **kwargs: [_task()])
    for _ in range(2):
        run_benchmark.main([
            "--benchmark", "coding",
            "--baseline", "no_memory",
            "--dry-run",
        ])

    run_roots = sorted(path for path in (tmp_path / "runs").iterdir() if path.is_dir())
    assert len(run_roots) == 2
    assert all(re.search(r"_\d{8}T\d{6}_[0-9a-f]{6}$", path.name) for path in run_roots)


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


def test_task_config_uses_coordinate_mode_key_and_wires_llm(monkeypatch):
    # regression: Config reads coordinate_mode; coordination_mode would crash
    monkeypatch.setenv("MARBLE_EVAL_MODEL", "fixed-eval-model")
    t = _task()
    cfg = task_config(t, "heuristic", llm="gpt-4o-mini")
    assert cfg["coordinate_mode"] == "graph"
    assert "coordination_mode" not in cfg
    assert cfg["llm"] == "gpt-4o-mini"
    # spec §6.4: evaluator model recorded, never empty
    assert cfg["metrics"]["evaluate_llm"] == "fixed-eval-model"


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


def test_cli_parameters_propagate_to_dry_run(monkeypatch, tmp_path):
    from marble.experiments import run_benchmark

    monkeypatch.setattr(
        run_benchmark,
        "load_tasks",
        lambda *args, **kwargs: [_task()],
    )
    run_benchmark.main([
        "--benchmark", "coding",
        "--baseline", "no_memory",
        "--dry-run",
        "--out", str(tmp_path),
        "--max-cards", "2",
        "--max-reads-per-step", "1",
        "--lambda", "0.1",
        "--beta", "0.5",
    ])
    summary = next(tmp_path.rglob("summary.json"))
    recorded = json.loads(summary.read_text())
    assert recorded["retrieval"]["max_cards"] == 2
    assert recorded["reward_config"] == {"lambda": 0.1, "beta": 0.5}


def test_manifest_seed_is_default_and_cli_seed_wins(monkeypatch, tmp_path):
    from marble.experiments import run_benchmark

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"seed": 42, "tasks": []}), encoding="utf-8")
    task = _task()
    captured = []
    monkeypatch.setattr(run_benchmark, "load_manifest_tasks", lambda *args, **kwargs: [task])
    monkeypatch.setattr(
        run_benchmark,
        "run_task",
        lambda task, baseline, out_root, **kwargs: captured.append(kwargs) or {"status": "dry_run", "task_id": task.task_id},
    )

    run_benchmark.main([
        "--benchmark", "coding", "--baseline", "no_memory", "--manifest",
        str(manifest), "--out", str(tmp_path / "default"), "--dry-run",
    ])
    assert captured[-1]["seed"] == 42

    run_benchmark.main([
        "--benchmark", "coding", "--baseline", "no_memory", "--manifest",
        str(manifest), "--seed", "7", "--out", str(tmp_path / "explicit"),
        "--dry-run",
    ])
    assert captured[-1]["seed"] == 7


def test_database_score_uses_explicit_prediction_overlap():
    from marble.experiments.run_benchmark import _score_result

    evaluator = type(
        "Evaluator",
        (),
        {
            "metrics": {
                "task_evaluation": {
                    "root_cause": ["LOCK_CONTENTION", "VACUUM"],
                    "predicted": "Final causes: LOCK_CONTENTION and VACUUM.",
                }
            }
        },
    )()
    score = _score_result("database", evaluator, "task", "result", None)

    assert score["score_status"] == "available"
    assert score["task_score"] == 1.0
    assert score["task_success"] is True


def test_database_score_does_not_award_nonempty_unrelated_output():
    from marble.experiments.run_benchmark import _score_result

    evaluator = type(
        "Evaluator",
        (),
        {
            "metrics": {
                "task_evaluation": {
                    "root_cause": ["LOCK_CONTENTION"],
                    "predicted": "The agents completed their investigation.",
                }
            }
        },
    )()
    score = _score_result("database", evaluator, "task", "result", None)

    assert score["task_score"] == 0.0
    assert score["task_success"] is False


def test_rating_score_rejects_invalid_ratings_and_tracks_agreement():
    from marble.experiments.run_benchmark import _score_result

    class Evaluator:
        metrics = {"task_evaluation": {}}

        def evaluate_task_world(self, task, result):
            self.metrics["task_evaluation"] = {
                "buyer": {"strategy": 5, "outcome": 4},
                "seller": {"strategy": 4, "outcome": 5},
            }

    environment = type("World", (), {"agreement_reached": True})()
    score = _score_result("bargaining", Evaluator(), "task", "summary", environment)
    assert score["score_status"] == "available"
    assert score["task_success"] is True

    class InvalidEvaluator:
        metrics = {"task_evaluation": {"x": -1}}

        def evaluate_task_research(self, task, result):
            pass

    assert _score_result(
        "research", InvalidEvaluator(), "task", "idea", None
    )["score_status"] == "unavailable"
