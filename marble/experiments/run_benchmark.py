"""Unified MultiAgentBench runner (spec §12).

Heavy engine/litellm imports stay lazy so the module (and its tests) load
offline; they happen only inside _run_real_episode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from marble.benchmarks import BENCHMARKS, BenchmarkTask, load_tasks
from marble.controllers import (
    AbsentController,
    GlobalAlwaysController,
    HeuristicController,
    LocalPolicyController,
    PrivateOnlyController,
)
from marble.experiments.ablations import (
    apply_to_task_config,
    controller_kwargs,
    parse_ablation,
    reward_override,
    wrap_controller,
)
from marble.experiments.engine_bridge import MemoryStep, build_governed_engine_cls
from marble.memory import GovernedMemory, MemoryBank, TraceLogger

BASELINES = (
    "no_memory",
    "global_always",
    "private_only",
    "heuristic",
    "learned_controller",
    "qwen_sft",
    "qwen_rl",
)

# ponytail: dataset llm is often ""; workers run through litellm, so a missing
# model string must not reach BaseAgent as "". Fall back to an env-overridable
# default; if still empty, _require_worker_model errors clearly.
DEFAULT_WORKER_MODEL = "gpt-3.5-turbo"

_WORKER_KEY_VARS = ("OPENAI_API_KEY", "NVAPI_KEY", "MARBLE_API_KEY")


# ----------------------------------------------------------------- controllers
def make_controller(
    baseline: str,
    controller_checkpoint: Optional[str] = None,
    ablation: Optional[str] = None,
    *,
    qwen_base_model: Optional[str] = None,
    qwen_api_base: Optional[str] = None,
    qwen_api_key: Optional[str] = None,
    qwen_api_model: Optional[str] = None,
    qwen_temperature: float = 0.0,
) -> Any:
    controller: Any = None
    if baseline == "no_memory":
        controller = AbsentController()
    elif baseline == "global_always":
        controller = GlobalAlwaysController()
    elif baseline == "private_only":
        controller = PrivateOnlyController()
    elif baseline == "heuristic":
        controller = HeuristicController()
    elif baseline == "learned_controller":
        # unified learned controller: LocalPolicyController (linear policy).
        # Untrained (no checkpoint) -> argmax over zero weights -> absent.
        # With checkpoint -> loaded policy. Same component in Stage B and Stage E
        # so the two are comparable (review: was JsonController vs LocalPolicy).
        controller = (
            LocalPolicyController.load(controller_checkpoint)
            if controller_checkpoint
            else LocalPolicyController()
        )
    elif baseline in ("qwen_sft", "qwen_rl"):
        from marble.controllers.qwen_lora import (
            api_generate_fn,
            local_generate_fn,
            make_qwen_lora_controller,
        )

        base_model = (
            qwen_base_model
            or os.environ.get("MARBLE_QWEN_BASE_MODEL")
            or "Qwen/Qwen3-4B-Instruct-2507"
        )
        api_base = qwen_api_base or os.environ.get("MARBLE_QWEN_API_BASE")
        if api_base:
            api_key = (
                qwen_api_key
                or os.environ.get("MARBLE_QWEN_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
                or os.environ.get("NVAPI_KEY")
                or os.environ.get("MARBLE_API_KEY")
            )
            if not api_key:
                raise RuntimeError(
                    "Qwen endpoint mode needs --qwen-api-key or MARBLE_QWEN_API_KEY"
                )
            generate_fn = api_generate_fn(
                api_base,
                api_key,
                qwen_api_model
                or os.environ.get("MARBLE_QWEN_API_MODEL")
                or base_model,
            )
        else:
            if not controller_checkpoint:
                raise ValueError(
                    f"{baseline} local runtime needs --controller-checkpoint "
                    "pointing to a LoRA adapter, or --qwen-api-base"
                )
            generate_fn = local_generate_fn(
                base_model, controller_checkpoint, temperature=qwen_temperature
            )
        controller = make_qwen_lora_controller(generate_fn)
    else:
        raise ValueError(f"unknown baseline {baseline!r}; choose from {BASELINES}")
    if ablation:
        factor, option = parse_ablation(ablation)
        controller = wrap_controller(controller, factor, option)
    return controller


def make_governed(
    baseline: str,
    trace_path: str | Path,
    controller_checkpoint: Optional[str] = None,
    ablation: Optional[str] = None,
    **qwen_runtime: Optional[str],
) -> GovernedMemory:
    return GovernedMemory(
        MemoryBank(),
        make_controller(
            baseline,
            controller_checkpoint,
            ablation,
            **qwen_runtime,
        ),
        trace=TraceLogger(str(trace_path)),
    )


# --------------------------------------------------------------------- config
def task_config(
    task: BenchmarkTask,
    baseline: str,
    max_cards: int = 6,
    max_reads_per_step: int = 2,
    retriever: str = "key_first",
    llm: str = "",
    ablation: Optional[str] = None,
) -> Dict[str, Any]:
    """Original record + governed memory block injected (source untouched)."""
    # ponytail: deployment pins worker via MARBLE_WORKER_MODEL env; that MUST
    # override per-dataset llm (e.g. minecraft ships "gpt-4o-mini").
    worker = llm or os.environ.get("MARBLE_WORKER_MODEL", "") or task.llm or DEFAULT_WORKER_MODEL
    cfg: Dict[str, Any] = {
        "task": {"content": task.task},
        "agents": [dict(a) for a in task.agents],
        "relationships": [list(r) for r in task.relationships],
        "environment": dict(task.environment),
        "memory": dict(task.memory),
        "metrics": dict(task.metrics),
        "engine_planner": dict(task.engine_planner),
        "output": dict(task.output),
        # ponytail: Config reads coordinate_mode (config.py), not coordination_mode
        "coordinate_mode": "graph",
        # agents fall back to config.llm; never empty (litellm rejects "").
        "llm": worker,
        # spec §6.4: evaluator must use a fixed non-empty model, never ""
        "metrics": {
            **dict(task.metrics),
            "evaluate_llm": (
                os.environ.get("MARBLE_EVAL_MODEL")
                or task.metrics.get("evaluate_llm")
                or worker
            ),
        },
    }
    # ponytail: JSONL agent configs may carry a per-agent llm (e.g. bargaining
    # ships "gpt-4o"); Engine reads agent_config.get('llm', config.llm) so that
    # field OVERRIDES our resolved worker. Force all agents to the deployment
    # worker so a single pinned model serves the whole episode.
    for _agent in cfg["agents"]:
        _agent["llm"] = worker
    # ponytail: original JSONL leaves env type/max_iterations empty; fill so
    # Engine.__init__ does not raise on an unsupported empty type
    _ENV_DEFAULTS = {"coding": "Coding", "research": "Research", "database": "DB",
                     "bargaining": "Web", "minecraft": "Minecraft"}
    env = cfg["environment"]
    # ponytail: JSONL ships type="" (falsy) — setdefault won't override it, so
    # check falsy explicitly or Engine raises "Unsupported environment type"
    if not str(env.get("type", "")).strip():
        env["type"] = _ENV_DEFAULTS.get(task.benchmark, "Base")
    if not str(env.get("max_iterations", "")).strip():
        env["max_iterations"] = 10
    if baseline != "no_memory":
        cfg["memory"] = {
            **cfg["memory"],
            "backend": "governed",
            "controller": baseline,
            "retriever": retriever,
            "max_cards": max_cards,
            "max_reads_per_step": max_reads_per_step,
        }
    if ablation:
        from marble.experiments.ablations import apply_to_task_config, parse_ablation

        factor, option = parse_ablation(ablation)
        cfg = apply_to_task_config(cfg, factor, option)
    return cfg


def _write_config(path: Path, cfg: Dict[str, Any]) -> None:
    try:
        import yaml

        text = yaml.safe_dump(cfg, allow_unicode=True)
    except ImportError:  # JSON is valid YAML; Config.load reads either
        text = json.dumps(cfg, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")


# ------------------------------------------------------------------- episodes
def plan_runs(
    baselines: List[str],
    tasks: List[BenchmarkTask],
) -> List[tuple]:
    """Expand 'multi' into every fixed+learned baseline."""
    expanded: List[str] = []
    for b in baselines:
        expanded += list(BASELINES) if b == "multi" else [b]
    seen: List[str] = []
    for b in expanded:
        if b not in seen:
            seen.append(b)
    return [(b, t) for b in seen for t in tasks]


def run_task(
    task: BenchmarkTask,
    baseline: str,
    out_root: str | Path,
    *,
    dry_run: bool = False,
    seed: Optional[int] = None,
    max_iterations: Optional[int] = None,
    max_cards: int = 6,
    max_reads_per_step: int = 2,
    retriever: str = "key_first",
    llm: str = "",
    ablation: Optional[str] = None,
    controller_checkpoint: Optional[str] = None,
    qwen_base_model: Optional[str] = None,
    qwen_api_base: Optional[str] = None,
    qwen_api_key: Optional[str] = None,
    qwen_api_model: Optional[str] = None,
    qwen_temperature: float = 0.0,
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
    tdir = Path(out_root) / baseline / task.benchmark / str(task.task_id)
    tdir.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []

    cfg = task_config(
        task, baseline, max_cards, max_reads_per_step, retriever, llm, ablation
    )
    # ponytail: ablations mutate the config memory block (policy->controller name,
    # retrieval:none->max_cards=0). make_controller is keyed by baseline, so the
    # runner must read the *effective* controller/max_cards back out of config.
    effective_baseline = baseline
    effective_max_cards = max_cards
    mem_block = cfg.get("memory")
    if baseline != "no_memory" and isinstance(mem_block, dict):
        effective_baseline = mem_block.get("controller", baseline)
        if "max_cards" in mem_block:
            effective_max_cards = mem_block["max_cards"]
    if max_iterations is not None:
        cfg["environment"]["max_iterations"] = max_iterations
    # ponytail: Engine opens output.file_path; leave empty -> open("") IOError.
    # Absolute: engine chdir's into marble/ for evaluator prompts, breaking relative paths.
    if not str(cfg["output"].get("file_path", "")).strip():
        cfg["output"]["file_path"] = str((tdir / "output.json").resolve())
    _write_config(tdir / "config.yaml", cfg)

    summary: Dict[str, Any] = {
        "method": baseline,
        "benchmark": task.benchmark,
        "task_id": task.task_id,
        "seed": seed,
        "status": "ok",
        # spec §11.2/§11.3: record the exact models used, all non-empty
        "worker_model": llm or os.environ.get("MARBLE_WORKER_MODEL", "") or task.llm or DEFAULT_WORKER_MODEL,
        "controller_model": (
            "learned_controller(local_policy)"
            if baseline == "learned_controller"
            else (
                (
                    "qwen_rl(policy_gradient;"
                    f"{qwen_api_model or qwen_base_model or os.environ.get('MARBLE_QWEN_BASE_MODEL') or 'Qwen/Qwen3-4B-Instruct-2507'})"
                    if baseline == "qwen_rl"
                    else f"qwen_sft({qwen_api_model or qwen_base_model or os.environ.get('MARBLE_QWEN_BASE_MODEL') or 'Qwen/Qwen3-4B-Instruct-2507'})"
                )
                if baseline in ("qwen_sft", "qwen_rl")
                else baseline
            )
        ),
        "evaluator_model": os.environ.get("MARBLE_EVAL_MODEL", "") or cfg["metrics"].get("evaluate_llm", ""),
    }

    if dry_run:
        summary["status"] = "dry_run"
        (tdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # spec §10/§11.9: never overwrite a completed episode on re-run; resume skips it
    completed = tdir / "summary.json"
    if completed.exists():
        try:
            if json.loads(completed.read_text()).get("status") == "ok":
                return json.loads(completed.read_text())
        except Exception:
            pass

    try:
        metrics = _run_real_episode(
            task, effective_baseline, tdir, cfg,
            max_cards=effective_max_cards,
            max_reads_per_step=max_reads_per_step,
            controller_checkpoint=controller_checkpoint,
            llm=llm,
            ablation=ablation,
            retriever=retriever,
            qwen_base_model=qwen_base_model,
            qwen_api_base=qwen_api_base,
            qwen_api_key=qwen_api_key,
            qwen_api_model=qwen_api_model,
            qwen_temperature=qwen_temperature,
        )
        summary.update(metrics)
        # spec §10: empty evaluator score -> mark, do not fake a zero
        if metrics.get("task_score", 0) == 0 and not (tdir / "output.json").exists():
            summary["status"] = "score_unavailable"
    except Exception as exc:  # noqa: BLE001 — one bad task must not kill the sweep
        summary["status"] = "error"
        errors.append(f"{type(exc).__name__}: {exc}")
        errors.append(traceback.format_exc())

    if errors:
        (tdir / "errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")
    (tdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _require_worker_key() -> None:
    import os

    if not any(os.environ.get(k) for k in _WORKER_KEY_VARS):
        raise RuntimeError(
            "real rollout needs a worker API key: set one of "
            f"{', '.join(_WORKER_KEY_VARS)} (or use --dry-run)"
        )


def _run_real_episode(
    task: BenchmarkTask,
    baseline: str,
    tdir: Path,
    cfg: Dict[str, Any],
    *,
    max_cards: int,
    max_reads_per_step: int,
    controller_checkpoint: Optional[str],
    llm: str = "",
    ablation: Optional[str] = None,
    retriever: str = "key_first",
    qwen_base_model: Optional[str] = None,
    qwen_api_base: Optional[str] = None,
    qwen_api_key: Optional[str] = None,
    qwen_api_model: Optional[str] = None,
    qwen_temperature: float = 0.0,
) -> Dict[str, Any]:
    _require_worker_key()
    import os

    from marble.configs.config import Config
    from marble.memory.rewards import proposal_rewards

    mem = make_governed(
        baseline,
        # absolute: the engine chdirs into marble/ mid-run; a relative path breaks
        str((tdir / "memory_trace.jsonl").resolve()),
        controller_checkpoint,
        ablation,
        qwen_base_model=qwen_base_model,
        qwen_api_base=qwen_api_base,
        qwen_api_key=qwen_api_key,
        qwen_api_model=qwen_api_model,
        qwen_temperature=qwen_temperature,
    )
    # ponytail: selector "top" makes agents read the top-ranked cards without an
    # extra API call, so memory actually influences the task
    if retriever not in ("key_first", "none"):
        print(f"[warn] retrieval {retriever!r} not implemented; using key_first")
    eff_max_cards = 0 if retriever == "none" else max_cards
    agent_role_map = {
        str(agent["agent_id"]): " | ".join(
            str(agent.get(field, "")).strip()
            for field in ("type", "profile")
            if str(agent.get(field, "")).strip()
        )
        for agent in task.agents
        if agent.get("agent_id")
    }
    harness = MemoryStep(
        mem, max_cards=eff_max_cards, max_reads_per_step=max_reads_per_step,
        selector="top", task_goal=task.task, agent_role_map=agent_role_map,
    )
    harness.task_id = str(task.task_id)

    EngineCls = build_governed_engine_cls()
    # harness must exist BEFORE Engine.__init__ (agents read it during _initialize_agents)
    EngineCls.memory_harness = harness

    config = Config.load(str(tdir / "config.yaml"))
    # Evaluator opens a RELATIVE 'evaluator/evaluator_prompts.json'; MARBLE expects
    # cwd = marble/ — chdir for the engine run, restore after.
    marble_dir = Path(__file__).resolve().parents[1]
    prev_cwd = os.getcwd()
    os.chdir(marble_dir)
    try:
        engine = EngineCls(config)
        engine.start()
        # spec §10: persist rejected controller outputs (redacted: no keys/prompts)
        controller = getattr(mem, "controller", None)
        if controller is not None and hasattr(controller, "rejections") and controller.rejections:
            dbg = [
                {"proposal_id": r.get("proposal_id"), "reason": r.get("reason"),
                 "raw": r.get("raw")}
                for r in controller.rejections
            ]
            (tdir / "controller_debug.jsonl").resolve().write_text(
                "\n".join(json.dumps(x) for x in dbg) + "\n", encoding="utf-8")
        ev = getattr(engine, "evaluator", None)
        if ev is not None and hasattr(ev, "update"):
            try:
                ev.update(engine.environment, engine.agents)
            except Exception as exc:  # noqa: BLE001 — never let scoring kill the run
                print(f"[warn] evaluator.update failed: {exc}")
    finally:
        os.chdir(prev_cwd)

    metrics: Dict[str, Any] = {}
    evaluator = getattr(engine, "evaluator", None)
    evaluator_metrics = getattr(evaluator, "metrics", None)
    if isinstance(evaluator_metrics, dict):
        result_text = ""
        out_path = Path(cfg["output"].get("file_path", ""))
        if out_path.exists():
            result_text = out_path.read_text(encoding="utf-8", errors="ignore")
        # Evaluator has no 'task_score'; derive from benchmark-specific eval with
        # task_completion mean as a lower bound
        task_score = _benchmark_score(task.benchmark, evaluator, task.task, result_text)
        completions = evaluator_metrics.get("task_completion", [])
        if completions:
            task_score = max(task_score, sum(completions) / len(completions))
        metrics["task_score"] = task_score
        metrics["engine_metrics"] = {
            k: v for k, v in evaluator_metrics.items() if isinstance(v, (int, float, str))
        }

    events = _read_jsonl(tdir / "memory_trace.jsonl")
    reward_kw: Dict[str, float] = {}
    if ablation:
        factor, option = parse_ablation(ablation)
        reward_kw = reward_override(factor, option)
    credits = proposal_rewards(
        events, task_score=float(metrics.get("task_score", 0.0)),
        same_task_baseline=0.0, **reward_kw
    )
    (tdir / "reward.json").write_text(json.dumps(credits, indent=2), encoding="utf-8")
    return metrics


def _apply_split(tasks: List[BenchmarkTask], split: str) -> List[BenchmarkTask]:
    """Deterministic train/test split by task_id (spec §15 fairness)."""
    if split in ("all", ""):
        return tasks
    buckets: Dict[str, List[BenchmarkTask]] = {"train": [], "test": []}
    for t in tasks:
        # stable 80/20 split independent of load order
        h = int(hashlib.sha256(str(t.task_id).encode()).hexdigest(), 16) % 10
        buckets["train" if h < 8 else "test"].append(t)
    if split not in buckets:
        raise ValueError(f"unknown split {split!r}; choose all|train|test")
    return buckets[split]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------- scoring
def _benchmark_score(benchmark: str, evaluator, task_content: str,
                     result_text: str) -> float:
    """Normalized 0..1 task score from MARBLE's per-benchmark evaluator.

    ponytail: real scoring needs MARBLE's eval LLM (gpt-3.5-turbo default).
    We call the matching evaluate_* when available and normalize its output;
    when the eval LLM is absent we fall back to the task_completion mean.
    is_task_completed() compares to empty ground_truth so it is usually 0.
    """
    m = getattr(evaluator, "metrics", {}) or {}
    completions = m.get("task_completion", [])
    base = sum(completions) / len(completions) if completions else 0.0
    if not result_text:
        return base
    try:
        if benchmark == "research":
            evaluator.evaluate_task_research(task_content, result_text)
            te = m.get("task_evaluation") or {}
            vals = [v for v in te.values() if isinstance(v, (int, float))]
            return sum(vals) / len(vals) / 5.0 if vals else base
        elif benchmark == "minecraft":
            evaluator.evaluate_task_world(task_content, result_text)
            te = m.get("task_evaluation") or {}
            vals = []
            for side in ("buyer", "seller"):
                vals += [v for v in (te.get(side) or {}).values()
                         if isinstance(v, (int, float))]
            return sum(vals) / len(vals) / 5.0 if vals else base
        elif benchmark == "database":
            # use whatever the engine/environment already scored; don't overwrite
            # with empty lists (that would force 0 regardless of real result)
            te = m.get("task_evaluation") or {}
            rc = te.get("root_cause") or te.get("predicted")
            return 1.0 if rc else base
        elif benchmark == "coding":
            evaluator.evaluate_code_quality(task_content, result_text)
            cq = m.get("code_quality") or {}
            vals = [v for v in cq.values() if isinstance(v, (int, float))]
            return sum(vals) / len(vals) / 5.0 if vals else base
    except Exception as exc:  # noqa: BLE001 — eval LLM may be absent offline
        print(f"[warn] benchmark scoring failed ({benchmark}): {exc}")
    return base


# ------------------------------------------------------------------------ CLI
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run governed-memory baselines on MultiAgentBench")
    ap.add_argument("--benchmark", required=True,
                    help=f"one of {BENCHMARKS}, comma-separated, or 'all'")
    ap.add_argument("--baseline", default="heuristic",
                    help=f"one of {BASELINES} or 'multi'")
    ap.add_argument("--split", default="all", help="all|train|test (deterministic by task_id)")
    ap.add_argument("--task-ids", default="", help="comma-separated task ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--retrieval", default="key_first")
    ap.add_argument("--max-reads-per-step", type=int, default=2)
    ap.add_argument("--controller-checkpoint", default=None)
    ap.add_argument(
        "--qwen-base-model",
        default=None,
        help="base model path/name for qwen_sft/qwen_rl local adapter loading",
    )
    ap.add_argument(
        "--qwen-api-base",
        default=None,
        help="OpenAI-compatible endpoint for qwen_sft/qwen_rl; skips local loading",
    )
    ap.add_argument("--qwen-api-key", default=None, help="key for --qwen-api-base")
    ap.add_argument(
        "--qwen-api-model",
        default=None,
        help="served adapter model name for --qwen-api-base",
    )
    ap.add_argument(
        "--qwen-temperature",
        type=float,
        default=0.0,
        help="controller sampling temperature; use >0 for RL rollout collection",
    )
    ap.add_argument("--worker-model", default=None, help="litellm model string for workers")
    ap.add_argument("--ablation", default=None, help="factor:option, e.g. retrieval:none")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args(argv)

    # ponytail: global floor so any untimeouted raw call (arxiv fetch, requests) can't hang forever
    socket.setdefaulttimeout(int(os.environ.get("MARBLE_SOCKET_TIMEOUT", "120")))

    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()] or None
    bench_names = [b.strip() for b in args.benchmark.split(",") if b.strip()]
    if "all" in bench_names:
        bench_names = list(BENCHMARKS)
    tasks: List[BenchmarkTask] = []
    for bn in bench_names:
        if bn not in BENCHMARKS:
            raise ValueError(f"unknown benchmark {bn!r}; choose from {BENCHMARKS} or 'all'")
        tasks += load_tasks(bn, start=args.start, task_ids=task_ids)
    tasks = _apply_split(tasks, args.split)
    if args.limit:
        tasks = tasks[: args.limit]
    if args.worker_model:
        os.environ.setdefault("MARBLE_WORKER_MODEL", args.worker_model)

    # --baseline multi expands to all methods
    baselines = ["multi"] if args.baseline == "multi" else [args.baseline]
    runs = plan_runs(baselines, tasks)

    run_id = f"{args.benchmark}_{args.baseline}"
    if args.ablation:
        run_id += f"_ablation-{args.ablation.replace(':', '_')}"
    if args.seed is not None:
        run_id += f"_seed{args.seed}"
    out_root = Path(args.out) / run_id
    print(f"{len(runs)} episode(s) -> {out_root}")
    for baseline, task in runs:
        summary = run_task(
            task, baseline, out_root,
            dry_run=args.dry_run,
            seed=args.seed,
            max_iterations=args.max_iterations,
            max_reads_per_step=args.max_reads_per_step,
            retriever=args.retrieval,
            llm=args.worker_model or "",
            ablation=args.ablation,
            controller_checkpoint=args.controller_checkpoint,
            qwen_base_model=args.qwen_base_model,
            qwen_api_base=args.qwen_api_base,
            qwen_api_key=args.qwen_api_key,
            qwen_api_model=args.qwen_api_model,
            qwen_temperature=args.qwen_temperature,
        )
        print(f"  [{summary['status']}] {baseline} task={task.task_id}")


if __name__ == "__main__":
    main()
