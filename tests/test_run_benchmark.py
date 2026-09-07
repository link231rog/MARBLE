import json
import os
import re
from dataclasses import replace
import pytest

from marble.benchmarks import load_tasks
from marble.controllers import LocalPolicyController
from marble.controllers import qwen_lora
from marble.experiments.run_benchmark import (
    BASELINES,
    make_controller,
    make_governed,
    plan_runs,
    provider_config,
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
    monkeypatch.delenv("MARBLE_QWEN_API_BASE", raising=False)

    def fake_local(base_model, lora_dir, temperature=0.0):
        seen["base_model"] = base_model
        seen["lora_dir"] = lora_dir
        seen["temperature"] = temperature
        return lambda prompt: '{"visibility":"global","supersedes":null}'

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
    ).visibility == "global"


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


def test_qwen_endpoint_mode_fast_fails_on_local_adapter_directory(monkeypatch):
    monkeypatch.setenv("MARBLE_QWEN_API_BASE", "https://api.siliconflow.cn/v1")
    monkeypatch.setenv("MARBLE_QWEN_API_KEY", "test-key")
    with pytest.raises(ValueError, match="Fatal checkpoint conflict"):
        make_controller("qwen_sft", controller_checkpoint="/local/path/sft_adapter")


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


def test_provider_defaults_are_recorded_without_api_key(monkeypatch):
    for var in (
        "MARBLE_WORKER_MODEL",
        "MARBLE_EVAL_MODEL",
        "MARBLE_ZAI_MODEL",
        "MARBLE_ZAI_EVAL_MODEL",
        "ZAI_MODEL",
        "ZAI_EVAL_MODEL",
        "ZAI_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    t = _task()
    cfg = task_config(t, "no_memory", provider="zai")
    assert provider_config("zai")["base_url"] == "https://api.z.ai/api/paas/v4"
    assert cfg["provider"] == "zai"
    assert cfg["llm"] == "openai/glm-4.7-flash"
    assert cfg["metrics"]["evaluate_llm"] == "openai/glm-4.7-flash"
    assert "key" not in cfg


def test_empero_provider_uses_its_own_models_and_default_key(monkeypatch):
    for var in (
        "MARBLE_WORKER_MODEL",
        "MARBLE_EVAL_MODEL",
        "MARBLE_EMPERO_MODEL",
        "MARBLE_EMPERO_EVAL_MODEL",
        "EMPERO_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    t = replace(_task(), llm="dataset-model")
    monkeypatch.setenv("MARBLE_WORKER_MODEL", "legacy-worker")
    monkeypatch.setenv("MARBLE_EVAL_MODEL", "legacy-evaluator")

    cfg = task_config(t, "no_memory", provider="empero")

    assert provider_config("empero") == {
        "base_url": "https://free.empero.org/v1",
        "worker_model": "openai/glm-5.3-flash",
        "eval_model": "openai/glm-5.3-flash",
        "key": "free",
    }
    assert cfg["llm"] == "openai/glm-5.3-flash"
    assert cfg["metrics"]["evaluate_llm"] == "openai/glm-5.3-flash"


def test_provider_specific_models_and_explicit_worker_take_precedence(monkeypatch):
    t = _task()
    monkeypatch.setenv("MARBLE_EMPERO_MODEL", "provider-worker")
    monkeypatch.setenv("MARBLE_EMPERO_EVAL_MODEL", "provider-evaluator")

    cfg = task_config(t, "no_memory", provider="empero")
    explicit = task_config(t, "no_memory", llm="cli-worker", provider="empero")

    assert cfg["llm"] == "provider-worker"
    assert cfg["metrics"]["evaluate_llm"] == "provider-evaluator"
    assert explicit["llm"] == "cli-worker"


def test_cli_provider_and_explicit_worker_model_propagate(monkeypatch, tmp_path):
    from marble.experiments import run_benchmark

    monkeypatch.setattr(run_benchmark, "load_tasks", lambda *args, **kwargs: [_task()])
    captured = []
    monkeypatch.setattr(
        run_benchmark,
        "run_task",
        lambda task, baseline, out_root, **kwargs: captured.append(kwargs)
        or {"status": "dry_run", "task_id": task.task_id},
    )
    run_benchmark.main([
        "--benchmark", "coding",
        "--baseline", "no_memory",
        "--provider", "zai",
        "--worker-model", "custom-worker",
        "--dry-run",
        "--out", str(tmp_path),
    ])
    assert captured[-1]["provider"] == "zai"
    assert captured[-1]["llm"] == "custom-worker"
    assert os.environ["OPENAI_API_BASE"] == "https://api.z.ai/api/paas/v4"


def test_cli_accepts_empero_provider(monkeypatch, tmp_path):
    from marble.experiments import run_benchmark

    monkeypatch.setattr(run_benchmark, "load_tasks", lambda *args, **kwargs: [_task()])
    captured = []
    monkeypatch.setattr(
        run_benchmark,
        "run_task",
        lambda task, baseline, out_root, **kwargs: captured.append(kwargs)
        or {"status": "dry_run", "task_id": task.task_id},
    )
    run_benchmark.main([
        "--benchmark", "coding",
        "--baseline", "no_memory",
        "--provider", "empero",
        "--dry-run",
        "--out", str(tmp_path),
    ])

    assert captured[-1]["provider"] == "empero"
    assert os.environ["OPENAI_API_BASE"] == "https://free.empero.org/v1"
    assert os.environ["OPENAI_API_KEY"] == "free"


def test_single_agent_task_config_keeps_only_one_agent():
    t = _task()
    cfg = task_config(t, "single_agent")

    assert len(cfg["agents"]) == 1
    chosen_id = cfg["agents"][0]["agent_id"]
    assert cfg["relationships"] == []
    assert "backend" not in cfg["memory"]


def test_task_config_overrides_empty_env_type():
    # JSONL ships environment.type="" which would make Engine raise
    # "Unsupported environment type"; task_config must fill the benchmark default.
    t = _task()
    assert not str(t.environment.get("type", "")).strip()
    cfg = task_config(t, "heuristic")
    assert cfg["environment"]["type"] == "Coding"
    assert cfg["environment"]["max_iterations"] in (5, 10)


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


def test_task_timeout_records_timeout_status_and_allows_rerun(tmp_path, monkeypatch):
    from marble.experiments.run_benchmark import TaskTimeoutError

    t = _task()

    def timeout_episode(*args, **kwargs):
        raise TaskTimeoutError("Task exceeded timeout limit of 5.0s")

    monkeypatch.setattr(
        "marble.experiments.run_benchmark._run_real_episode",
        timeout_episode,
    )
    summary = run_task(t, "heuristic", tmp_path, task_timeout=5.0)
    assert summary["status"] == "timeout"
    tdir = tmp_path / "heuristic" / "coding" / str(t.task_id)
    assert (tdir / "errors.log").exists()
    assert "TaskTimeoutError" in (tdir / "errors.log").read_text()

    events = [
        json.loads(line)
        for line in (tmp_path / "run_events.jsonl").read_text().splitlines()
    ]
    event_names = [e["event"] for e in events]
    assert "task_timeout" in event_names
    assert events[-1]["status"] == "timeout"

    # Verify resume does not skip timed-out tasks
    called = []
    def record_call(*args, **kwargs):
        called.append(True)
        return {"score_status": "available", "task_score": 1.0}

    monkeypatch.setattr(
        "marble.experiments.run_benchmark._run_real_episode",
        record_call,
    )
    summary2 = run_task(t, "heuristic", tmp_path, task_timeout=5.0)
    assert len(called) == 1
    assert summary2["status"] == "ok"


def test_no_proxy_with_ipv6_port_is_cleared(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,:1")
    monkeypatch.setenv("no_proxy", "::1,localhost")
    t = _task()
    run_task(t, "heuristic", tmp_path, dry_run=True)
    assert "NO_PROXY" not in os.environ
    assert "no_proxy" not in os.environ


def test_database_fixture_paths_isolated(tmp_path, monkeypatch):
    t = replace(_task(), benchmark="database")
    run_task(t, "heuristic", tmp_path, dry_run=True)
    tdir = tmp_path / "heuristic" / "database" / str(t.task_id)
    assert os.environ.get("MARBLE_DATASET_LOG") == str((tdir / "dataset.txt").resolve())
    assert os.environ.get("MARBLE_BADSQL_LOG") == str((tdir / "badsql.txt").resolve())


def test_visible_k_retriever_accepted_without_warning(capsys, tmp_path, monkeypatch):
    t = _task()
    summary = run_task(t, "heuristic", tmp_path, retriever="visible_k", dry_run=True)
    captured = capsys.readouterr()
    assert "[warn] retrieval 'visible_k' not implemented" not in captured.out
    assert summary["retrieval"]["name"] == "visible_k"


def test_summary_resume_config_mismatch_reruns(tmp_path, monkeypatch):
    t = _task()
    tdir = tmp_path / "heuristic" / "coding" / str(t.task_id)
    tdir.mkdir(parents=True)
    # Write a summary with max_cards=6
    stale_summary = {
        "status": "ok",
        "method": "heuristic",
        "task_id": t.task_id,
        "seed": 42,
        "setting": "method=heuristic|ablation=none|comm_gov=off|retrieval=key_first|max_cards=6|max_reads_per_step=2|lambda=0.05|beta=0.25",
    }
    (tdir / "summary.json").write_text(json.dumps(stale_summary))

    called = []
    def mock_run_real_episode(*args, **kwargs):
        called.append(True)
        return {"score_status": "available", "task_score": 1.0}

    monkeypatch.setattr("marble.experiments.run_benchmark._run_real_episode", mock_run_real_episode)
    # When running with max_cards=5 (the frozen default), the setting mismatches!
    summary = run_task(t, "heuristic", tmp_path, seed=42, max_cards=5)
    # It must re-run rather than return the stale max_cards=6 summary
    assert len(called) == 1
    assert summary["retrieval"]["max_cards"] == 5


def test_default_max_cards_is_five_across_entrypoints():
    from marble.experiments.engine_bridge import MemoryStep
    from marble.experiments.run_benchmark import _build_arg_parser
    from marble.memory.governed_memory import GovernedMemory
    import inspect

    # 1. Engine bridge MemoryStep default
    step = MemoryStep(None)
    assert step.max_cards == 5, f"MemoryStep default must be 5, got {step.max_cards}"

    # 2. CLI argument parser default
    parser = _build_arg_parser()
    args = parser.parse_args(["--benchmark", "database"])
    assert args.max_cards == 5, f"CLI --max-cards default must be 5, got {args.max_cards}"

    # 3. GovernedMemory.visible_keys signature default
    sig = inspect.signature(GovernedMemory.visible_keys)
    assert sig.parameters["top_k"].default == 5, f"GovernedMemory top_k default must be 5, got {sig.parameters['top_k'].default}"


def test_describe_controller_truthful_identity():
    from marble.experiments.run_benchmark import describe_controller

    # R07: .json checkpoint must be identified as local_policy linear even if baseline is ours_rl
    desc_json = describe_controller("ours_rl", controller_checkpoint="runs/ours_rl_policy.json")
    assert desc_json == "local_policy(linear;ours_rl_policy.json)"

    # LoRA checkpoint directory must be identified as qwen_rl with adapter
    desc_lora = describe_controller("ours_rl", controller_checkpoint="runs/qwen_adapter")
    assert "qwen_rl" in desc_lora and "adapter=qwen_adapter" in desc_lora

    # Standard baselines return their canonical name
    assert describe_controller("heuristic") == "heuristic"
    assert describe_controller("global_add_all") == "global_add_all"


def test_make_controller_rejects_json_for_qwen_baselines():
    import pytest

    with pytest.raises(ValueError, match="expects a Qwen LoRA adapter directory"):
        make_controller("ours_rl", controller_checkpoint="runs/ours_rl_policy.json")

    with pytest.raises(ValueError, match="expects a Qwen LoRA adapter directory"):
        make_controller("ours_sft", controller_checkpoint="runs/ours_sft_policy.json")


def test_baseline_aliases_for_linear_policy(tmp_path):
    p = tmp_path / "policy.json"
    LocalPolicyController().save(str(p))

    for name in ("local_policy", "linear_policy", "ours_linear_sft", "ours_linear_rl"):
        ctrl = make_controller(name, controller_checkpoint=str(p))
        assert isinstance(ctrl, LocalPolicyController)



