"""Unified MultiAgentBench runner (spec §12).

Heavy engine/litellm imports stay lazy so the module (and its tests) load
offline; they happen only inside _run_real_episode.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from marble.benchmarks import BENCHMARKS, BenchmarkTask, load_tasks
from marble.controllers import (
    AbsentController,
    GlobalAlwaysController,
    HeuristicController,
    JsonController,
    LocalPolicyController,
    PrivateOnlyController,
)
from marble.experiments.engine_bridge import MemoryStep, build_governed_engine_cls
from marble.memory import GovernedMemory, MemoryBank, TraceLogger

BASELINES = ("no_memory", "global_always", "private_only", "heuristic", "learned_controller")

_WORKER_KEY_VARS = ("OPENAI_API_KEY", "NVAPI_KEY", "MARBLE_API_KEY")


# ----------------------------------------------------------------- controllers
def make_controller(
    baseline: str,
    controller_checkpoint: Optional[str] = None,
) -> Any:
    if baseline == "no_memory":
        return AbsentController()
    if baseline == "global_always":
        return GlobalAlwaysController()
    if baseline == "private_only":
        return PrivateOnlyController()
    if baseline == "heuristic":
        return HeuristicController()
    if baseline == "learned_controller":
        if controller_checkpoint:
            return LocalPolicyController.load(controller_checkpoint)
        from marble.experiments.coding_rollout import NvidiaLLM

        llm = NvidiaLLM()  # raises RuntimeError when NVAPI_KEY missing
        return JsonController(
            lambda p: llm.act(
                "You are a strict memory-governance controller. Output ONLY the JSON decision.",
                p,
            )
        )
    raise ValueError(f"unknown baseline {baseline!r}; choose from {BASELINES}")


def make_governed(
    baseline: str,
    trace_path: str | Path,
    controller_checkpoint: Optional[str] = None,
) -> GovernedMemory:
    return GovernedMemory(
        MemoryBank(),
        make_controller(baseline, controller_checkpoint),
        trace=TraceLogger(str(trace_path)),
    )


# --------------------------------------------------------------------- config
def task_config(
    task: BenchmarkTask,
    baseline: str,
    max_cards: int = 6,
    max_reads_per_step: int = 2,
    retriever: str = "key_first",
) -> Dict[str, Any]:
    """Original record + governed memory block injected (source untouched)."""
    cfg: Dict[str, Any] = {
        "task": {"content": task.task},
        "agents": [dict(a) for a in task.agents],
        "relationships": [list(r) for r in task.relationships],
        "environment": dict(task.environment),
        "memory": dict(task.memory),
        "metrics": dict(task.metrics),
        "engine_planner": dict(task.engine_planner),
        "output": dict(task.output),
        "coordination_mode": "graph",
    }
    if baseline != "no_memory":
        cfg["memory"] = {
            **cfg["memory"],
            "backend": "governed",
            "controller": baseline,
            "retriever": retriever,
            "max_cards": max_cards,
            "max_reads_per_step": max_reads_per_step,
        }
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
    controller_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    if seed is not None:
        random.seed(seed)
    tdir = Path(out_root) / baseline / task.benchmark / str(task.task_id)
    tdir.mkdir(parents=True, exist_ok=True)
    errors: List[str] = []

    cfg = task_config(task, baseline, max_cards, max_reads_per_step, retriever)
    if max_iterations is not None:
        cfg["environment"]["max_iterations"] = max_iterations
    _write_config(tdir / "config.yaml", cfg)

    summary: Dict[str, Any] = {
        "method": baseline,
        "benchmark": task.benchmark,
        "task_id": task.task_id,
        "seed": seed,
        "status": "ok",
    }

    if dry_run:
        summary["status"] = "dry_run"
        (tdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    try:
        metrics = _run_real_episode(
            task, baseline, tdir, cfg,
            max_cards=max_cards,
            max_reads_per_step=max_reads_per_step,
            controller_checkpoint=controller_checkpoint,
        )
        summary.update(metrics)
    except Exception as exc:  # noqa: BLE001 — one bad task must not kill the sweep
        summary["status"] = "error"
        errors.append(f"{type(exc).__name__}: {exc}")

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
) -> Dict[str, Any]:
    _require_worker_key()
    # lazy: litellm chain lives behind these imports
    from marble.configs.config import Config

    config_path = tdir / "config.yaml"
    config = Config.load(str(config_path))
    mem = make_governed(baseline, tdir / "memory_trace.jsonl", controller_checkpoint)
    harness = MemoryStep(mem, max_cards=max_cards, max_reads_per_step=max_reads_per_step)
    harness.task_id = str(task.task_id)

    EngineCls = build_governed_engine_cls()
    engine = EngineCls(config)
    engine.memory_harness = harness
    engine.start()

    metrics: Dict[str, Any] = {}
    evaluator_metrics = getattr(getattr(engine, "evaluator", None), "metrics", None)
    if isinstance(evaluator_metrics, dict):
        metrics["task_score"] = evaluator_metrics.get("task_score", 0.0)
        metrics["engine_metrics"] = {
            k: v for k, v in evaluator_metrics.items() if isinstance(v, (int, float, str))
        }

    # per-proposal credits recomputable from trace + score (spec §17)
    events = _read_jsonl(tdir / "memory_trace.jsonl")
    from marble.memory.rewards import proposal_rewards

    credits = proposal_rewards(events, r_episode=float(metrics.get("task_score", 0.0)))
    (tdir / "reward.json").write_text(json.dumps(credits, indent=2), encoding="utf-8")
    return metrics


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------------ CLI
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run governed-memory baselines on MultiAgentBench")
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--baseline", default="heuristic",
                    help=f"one of {BASELINES} or 'multi'")
    ap.add_argument("--split", default="all")
    ap.add_argument("--task-ids", default="", help="comma-separated task ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--retrieval", default="key_first")
    ap.add_argument("--max-reads-per-step", type=int, default=2)
    ap.add_argument("--controller-checkpoint", default=None)
    ap.add_argument("--worker-model", default=None, help="litellm model string for workers")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args(argv)

    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()] or None
    tasks = load_tasks(args.benchmark, limit=args.limit, start=args.start, task_ids=task_ids)
    if args.worker_model:
        import os

        os.environ.setdefault("MARBLE_WORKER_MODEL", args.worker_model)

    # --baseline multi expands to all methods
    baselines = ["multi"] if args.baseline == "multi" else [args.baseline]
    runs = plan_runs(baselines, tasks)

    run_id = f"{args.benchmark}_{args.baseline}"
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
            controller_checkpoint=args.controller_checkpoint,
        )
        print(f"  [{summary['status']}] {baseline} task={task.task_id}")


if __name__ == "__main__":
    main()
