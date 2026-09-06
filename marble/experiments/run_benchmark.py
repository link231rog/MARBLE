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
import re
import secrets
import signal
import socket
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class TaskTimeoutError(TimeoutError):
    """Raised when an episode execution exceeds the allowed task timeout."""
    pass

from marble.benchmarks import BENCHMARKS, BenchmarkTask, load_tasks
from marble.controllers import (
    AbsentController,
    GlobalAlwaysController,
    HeuristicController,
    LTSStyleController,
    LocalPolicyController,
    PrivateOnlyController,
)
from marble.experiments.baselines import (
    MAIN_BASELINES,
    baseline_spec,
    canonical_baseline,
)
from marble.experiments.ablations import (
    apply_to_task_config,
    controller_kwargs,
    parse_ablation,
    reward_override,
    wrap_controller,
)
from marble.experiments.engine_bridge import MemoryStep, build_governed_engine_cls
from marble.experiments.task_manifest import load_manifest_tasks, read_manifest
from marble.memory import (
    GovernedMemory,
    MemoryBank,
    MemoryR1Adapter,
    MemoryR1Memory,
    TraceLogger,
)
from marble.memory.memory_r1_adapter import parse_crud_decision
from marble.memory.rewards import measured_memory_cost, token_count

# Kept as the public runner constant for existing callers/tests.
BASELINES = MAIN_BASELINES

# ponytail: dataset llm is often ""; workers run through litellm, so a missing
# model string must not reach BaseAgent as "". Fall back to an env-overridable
# default; if still empty, _require_worker_model errors clearly.
DEFAULT_WORKER_MODEL = "gpt-3.5-turbo"

_WORKER_KEY_VARS = ("OPENAI_API_KEY", "NVAPI_KEY", "MARBLE_API_KEY")
_PROVIDER_DEFAULTS = {
    "sensenova": {
        "base_url": "https://api.sensenova.cn/compatible-mode/v1",
        "worker_model": "openai/deepseek-v4-flash",
        "key_vars": ("SENSENOVA_API_KEY", "OPENAI_API_KEY", "MARBLE_API_KEY"),
    },
    "zai": {
        "base_url": "https://api.z.ai/api/paas/v4",
        "worker_model": "openai/glm-4.7-flash",
        "key_vars": ("ZAI_API_KEY", "OPENAI_API_KEY", "MARBLE_API_KEY"),
    },
    "empero": {
        "base_url": "https://free.empero.org/v1",
        "worker_model": "openai/glm-5.3-flash",
        "key_vars": ("EMPERO_API_KEY",),
        "default_key": "free",
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "worker_model": "openai/nvidia/nemotron-3-super-120b-a12b",
        "key_vars": ("NVIDIA_API_KEY", "NVAPI_KEY", "OPENAI_API_KEY", "MARBLE_API_KEY"),
    },
}


def provider_config(provider: str) -> Dict[str, Any]:
    if provider not in _PROVIDER_DEFAULTS:
        raise ValueError(f"unknown provider {provider!r}; choose sensenova|zai|empero|nvidia")
    defaults = _PROVIDER_DEFAULTS[provider]
    prefix = provider.upper()
    return {
        "base_url": os.environ.get(
            f"MARBLE_{prefix}_API_BASE",
            os.environ.get(f"{prefix}_API_BASE", defaults["base_url"]),
        ),
        "worker_model": os.environ.get(
            f"MARBLE_{prefix}_MODEL",
            os.environ.get(f"{prefix}_MODEL", defaults["worker_model"]),
        ),
        "eval_model": os.environ.get(
            f"MARBLE_{prefix}_EVAL_MODEL",
            os.environ.get(
                f"{prefix}_EVAL_MODEL",
                defaults["worker_model"],
            ),
        ),
        "key": next(
            (os.environ[name] for name in defaults["key_vars"] if os.environ.get(name)),
            defaults.get("default_key", ""),
        ),
    }


def _configure_provider(provider: str) -> Dict[str, Any]:
    config = provider_config(provider)
    os.environ["OPENAI_API_BASE"] = config["base_url"]
    if config["key"]:
        os.environ["OPENAI_API_KEY"] = config["key"]
    return config


# ----------------------------------------------------------------- controllers
def make_controller(
    baseline: str,
    controller_checkpoint: Optional[str] = None,
    ablation: Optional[str] = None,
    *,
    worker_model: Optional[str] = None,
    qwen_base_model: Optional[str] = None,
    qwen_api_base: Optional[str] = None,
    qwen_api_key: Optional[str] = None,
    qwen_api_model: Optional[str] = None,
    qwen_temperature: float = 0.0,
    **extra_kwargs: Any,
) -> Any:
    baseline = canonical_baseline(baseline)
    spec = baseline_spec(baseline)
    if baseline == "ours_private_to_global" and not ablation:
        ablation = "visibility:private_to_global"
    controller: Any = None
    if spec.controller == "absent":
        controller = AbsentController()
    elif spec.controller == "global_add_all":
        controller = GlobalAlwaysController()
    elif spec.controller == "private_only":
        controller = PrivateOnlyController()
    elif spec.controller == "heuristic":
        controller = HeuristicController()
    elif spec.controller == "lts_binary":
        judge_fn = _make_lts_judge(worker_model) if worker_model else None
        controller = LTSStyleController(judge_fn=judge_fn)
    elif spec.controller == "local_policy":
        # unified learned controller: LocalPolicyController (linear policy).
        # Untrained (no checkpoint) -> argmax over zero weights -> absent.
        # With checkpoint -> loaded policy. Same component in Stage B and Stage E
        # so the two are comparable (review: was JsonController vs LocalPolicy).
        controller = (
            LocalPolicyController.load(controller_checkpoint)
            if controller_checkpoint
            else LocalPolicyController()
        )
    elif spec.controller in ("qwen_sft", "qwen_rl"):
        if controller_checkpoint and str(controller_checkpoint).endswith(".json"):
            controller = LocalPolicyController.load(controller_checkpoint)
        else:
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
            # Pass ablation kwargs (drop_fields) so input/schema ablations
            # actually change the Qwen controller prompt.
            qwen_kw: Dict[str, Any] = {}
            if ablation:
                _f, _o = parse_ablation(ablation)
                qwen_kw = controller_kwargs(_f, _o)
            controller = make_qwen_lora_controller(generate_fn, **qwen_kw)
    elif spec.controller == "memory_r1_crud":
        # R1's manager is a storage runtime selected by make_memory_runtime().
        controller = None
    else:  # pragma: no cover - registry validation makes this unreachable
        raise AssertionError(f"unhandled controller type {spec.controller!r}")
    if ablation:
        factor, option = parse_ablation(ablation)
        controller = wrap_controller(controller, factor, option)
    return controller


def describe_controller(
    baseline: str,
    controller_checkpoint: Optional[str] = None,
    qwen_base_model: Optional[str] = None,
    qwen_api_model: Optional[str] = None,
    ablation: Optional[str] = None,
) -> str:
    """Accurately identify controller type and backend without mislabeling (spec §12.3 R07)."""
    canonical = canonical_baseline(baseline)
    spec = baseline_spec(canonical)
    desc: str
    if spec.controller in ("local_policy",):
        ckpt_name = Path(controller_checkpoint).name if controller_checkpoint else "untrained"
        desc = f"local_policy(linear;{ckpt_name})"
    elif spec.controller in ("qwen_sft", "qwen_rl"):
        if controller_checkpoint and str(controller_checkpoint).endswith(".json"):
            desc = f"local_policy(linear;{Path(controller_checkpoint).name})"
        else:
            model_name = (
                qwen_api_model
                or qwen_base_model
                or os.environ.get("MARBLE_QWEN_BASE_MODEL")
                or "Qwen/Qwen3-4B-Instruct-2507"
            )
            adapter = Path(controller_checkpoint).name if controller_checkpoint else "none"
            prefix = "qwen_rl" if spec.controller == "qwen_rl" else "qwen_sft"
            desc = f"{prefix}(model={model_name};adapter={adapter})"
    else:
        desc = canonical

    if canonical == "ours_private_to_global" or (ablation and "private_to_global" in ablation):
        desc = f"{desc}[ablation=private_to_global]"
    return desc


def make_governed(
    baseline: str,
    trace_path: str | Path,
    controller_checkpoint: Optional[str] = None,
    ablation: Optional[str] = None,
    **qwen_runtime: Optional[str],
) -> GovernedMemory:
    worker_model = qwen_runtime.pop("worker_model", None)
    return GovernedMemory(
        MemoryBank(),
        make_controller(
            baseline,
            controller_checkpoint,
            ablation,
            worker_model=str(worker_model) if worker_model else None,
            **qwen_runtime,
        ),
        trace=TraceLogger(str(trace_path)),
    )


def make_memory_runtime(
    baseline: str,
    trace_path: str | Path,
    controller_checkpoint: Optional[str] = None,
    ablation: Optional[str] = None,
    **qwen_runtime: Optional[str],
) -> Any:
    """Build the storage runtime for governed, classical, or R1-style methods."""
    baseline = canonical_baseline(baseline)
    if baseline == "memory_r1_style":
        worker_model = str(qwen_runtime.pop("worker_model", "") or DEFAULT_WORKER_MODEL)
        return MemoryR1Adapter(
            memory=MemoryR1Memory(),
            trace=TraceLogger(str(trace_path)),
            manager=_make_r1_manager(worker_model),
            distill_fn=_make_r1_distiller(worker_model),
        )
    if baseline == "mem0_style":
        from marble.memory.classical_baselines import Mem0Adapter

        worker_model = str(qwen_runtime.pop("worker_model", "") or "")
        manager_fn = _make_mem0_manager(worker_model) if worker_model else None
        return Mem0Adapter(
            trace=TraceLogger(str(trace_path)),
            manager_fn=manager_fn,
        )
    if baseline == "amem_style":
        from marble.memory.classical_baselines import AMemAdapter

        return AMemAdapter(
            trace=TraceLogger(str(trace_path)),
        )
    if baseline == "memoryos_style":
        from marble.memory.classical_baselines import MemoryOSAdapter

        return MemoryOSAdapter(
            trace=TraceLogger(str(trace_path)),
        )
    if baseline == "g_memory_style":
        from marble.memory.mas_baselines import GMemoryAdapter

        return GMemoryAdapter(
            trace=TraceLogger(str(trace_path)),
        )
    if baseline == "collabmem_style":
        from marble.memory.mas_baselines import CollabMemAdapter

        return CollabMemAdapter(
            trace=TraceLogger(str(trace_path)),
        )
    if baseline == "copper_style":
        from marble.memory.mas_baselines import COPPERAdapter

        return COPPERAdapter(
            trace=TraceLogger(str(trace_path)),
        )
    return make_governed(
        baseline,
        trace_path,
        controller_checkpoint,
        ablation,
        **qwen_runtime,
    )



def _worker_completion(model: str, prompt: str, max_tokens: int) -> str:
    """Use the experiment worker model for R1's adapted manager/distiller and LTS judge."""
    from marble.llms.model_prompting import model_prompting

    try:
        response = model_prompting(
            llm_model=model,
            messages=[{"role": "user", "content": prompt}],
            return_num=1,
            max_token_num=max_tokens,
            temperature=0.0,
            top_p=None,
            stream=None,
        )[0]
        return str(getattr(response, "content", "") or "")
    except Exception:
        return ""


def _make_r1_manager(worker_model: str):
    """Memory-R1 CRUD Memory Manager (ACL 2026 Appendix C.1).

    Uses the exact verbatim instructions and 4-operation schema (ADD, UPDATE, DELETE, NONE/NOOP)
    specified in the original Memory-R1 paper.
    """
    def manager(proposal, active):
        index = [
            {"id": str(item.memory_id), "text": f"{item.title}: {item.raw_value[:256]}"}
            for item in active
        ]
        retrieved_fact = f"{proposal.title}: {proposal.raw_value[:512]}"
        prompt = (
            "You are a smart memory manager which controls the memory of a system. "
            "You can perform four operations: (1) add into the memory, (2) update the memory, "
            "(3) delete from the memory, and (4) no change. Based on the above four operations, "
            "the memory will change. Compare newly retrieved facts with the existing memory. "
            "For each new fact, decide whether to: - ADD: Add it to the memory as a new element "
            "- UPDATE: Update an existing memory element - DELETE: Delete an existing memory element "
            "- NONE: Make no change (if the fact is already present or irrelevant).\n\n"
            "1. **Add**: If the retrieved facts contain new information not present in the memory, "
            "then you have to add it by generating a new ID in the id field.\n"
            "2. **Update**: If the retrieved facts contain information that is already present in the memory "
            "but the information is totally different, then you have to update it. If the retrieved fact contains "
            "information that conveys the same thing as the memory, keep the version with more detail. "
            "Important: When updating, keep the same ID and preserve old_memory.\n"
            "3. **Delete**: If the retrieved facts contain information that contradicts the memory, delete it. "
            "When deleting, return the same IDs — do not generate new IDs.\n"
            "4. **No Change**: If the retrieved facts are already present, make no change.\n\n"
            f"Old Memory: {json.dumps(index, ensure_ascii=False)}\n"
            f"Retrieved facts: {json.dumps([retrieved_fact], ensure_ascii=False)}\n\n"
            'Return JSON only: {"event": "ADD|UPDATE|DELETE|NONE", "id": null|"existing id"}'
        )
        raw = _worker_completion(worker_model, prompt, max_tokens=128)
        decision = parse_crud_decision(raw, active)
        decision.update(
            manager_input_tokens=token_count(prompt),
            manager_output_tokens=token_count(raw),
            manager_api_calls=1,
        )
        return decision

    return manager


def _make_r1_distiller(worker_model: str):
    """Memory-R1 Answer Agent Distillation (ACL 2026 Appendix C.2).

    Instructs the model to select and distill only evidence useful for answering
    the task, respecting timestamps and factual fidelity.
    """
    def distill(task_text: str, notes: List[str]) -> str:
        prompt = (
            "You are an intelligent memory assistant tasked with retrieving accurate information "
            "from task memories.\n"
            "# CONTEXT: You have access to memories from agents in a collaborative task execution. "
            "These memories contain timestamped information that may be relevant to answering the question.\n"
            "# INSTRUCTIONS:\n"
            "1. Carefully analyze all provided memories.\n"
            "2. If the memories contain contradictory information, prioritize the most recent memory.\n"
            "3. Select memories you found that are useful for answering the questions, and output them.\n"
            "4. Distill and compact only evidence useful for answering the task. Do not add unsupported facts.\n"
            "Return plain concise notes.\n\n"
            f"Task: {task_text}\n"
            "Memories:\n" + "\n".join(f"- {note}" for note in notes)
        )
        return _worker_completion(worker_model, prompt, max_tokens=256)

    return distill


def _make_lts_judge(worker_model: str):
    """LTS-LLM admission judge (ICML 2026 §4.1).

    Prompted decision by a frozen LLM deciding whether an intermediate step
    produces reusable, globally useful findings that should be admitted into
    the shared memory bank or discarded.
    """
    from marble.controllers.heuristic import _SHARED_SIGNALS

    def judge(proposal: Any) -> bool:
        if os.environ.get("PYTEST_CURRENT_TEST"):
            return bool(_SHARED_SIGNALS.search(proposal.title))
        prompt = (
            "You are the LTS (Learning to Share) Admission Controller for a parallel multi-agent system. "
            "Decide whether the following intermediate step produces reusable, globally useful findings "
            "that should be admitted to the shared memory bank to prevent redundant computation across teams.\n"
            "- YES: Admit into shared memory (general finding, critical evidence, or reusable intermediate state)\n"
            "- NO: Discard (transient local scratchpad, redundant observation, or noise)\n\n"
            f"Step Summary: {proposal.title}\n"
            f"Step Output: {proposal.raw_value[:512]}\n\n"
            'Return JSON only: {"admit": true|false, "decision": "YES|NO"}'
        )
        raw = _worker_completion(worker_model, prompt, max_tokens=32)
        match = re.search(r"\"(?:admit|decision)\"\s*:\s*(\"?\w+\"?)", raw, re.IGNORECASE)
        if match:
            val = match.group(1).replace('"', "").lower()
            if val in {"true", "yes"}:
                return True
            if val in {"false", "no"}:
                return False
        raw_upper = raw.upper()
        if "YES" in raw_upper or "TRUE" in raw_upper:
            return True
        if "NO" in raw_upper or "FALSE" in raw_upper:
            return False
        return bool(_SHARED_SIGNALS.search(proposal.title))

    return judge


def _make_mem0_manager(worker_model: str):
    """In-context Mem0-style manager performing CRUD operations via worker prompt."""
    from marble.memory.memory_r1_adapter import parse_crud_decision

    def manager(proposal, active):
        index = [
            {"memory_id": item.memory_id, "title": item.title, "value": item.raw_value[:256]}
            for item in active
        ]
        prompt = (
            "You are the Mem0 memory manager. Analyze the incoming observation and existing memories. "
            "Choose ADD if it contains new facts, UPDATE if it refines an existing memory (specify its memory_id), "
            "DELETE if it contradicts an existing memory, or NOOP if it is redundant or irrelevant. "
            'Return JSON only: {"operation": "ADD|UPDATE|DELETE|NOOP", "memory_id": null|"existing_id"}.\n'
            f"Existing memories: {json.dumps(index, ensure_ascii=False)}\n"
            f"New observation: {proposal.title}: {proposal.raw_value[:512]}"
        )
        raw = _worker_completion(worker_model, prompt, max_tokens=64)
        decision = parse_crud_decision(raw, active)
        decision.update(
            manager_input_tokens=token_count(prompt),
            manager_output_tokens=token_count(raw),
            manager_api_calls=1,
        )
        return decision

    return manager


# --------------------------------------------------------------------- config
def task_config(
    task: BenchmarkTask,
    baseline: str,
    max_cards: int = 5,
    max_reads_per_step: int = 2,
    retriever: str = "key_first",
    llm: str = "",
    ablation: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Original record + governed memory block injected (source untouched)."""
    if max_cards < 0 or max_reads_per_step < 0:
        raise ValueError("max_cards and max_reads_per_step must be non-negative")
    baseline = canonical_baseline(baseline)
    spec = baseline_spec(baseline)
    provider_defaults = provider_config(provider) if provider else None
    # A selected provider owns its model choice; the legacy global path applies
    # only when --provider is omitted.
    worker = llm or (
        provider_defaults["worker_model"]
        if provider_defaults
        else os.environ.get("MARBLE_WORKER_MODEL", "") or task.llm
    ) or DEFAULT_WORKER_MODEL
    task_payload = dict(getattr(task, "task_data", None) or {})
    if "content" not in task_payload:
        task_payload["content"] = task.task
    cfg: Dict[str, Any] = {
        "provider": provider,
        "task": task_payload,
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
                provider_defaults["eval_model"]
                if provider_defaults
                else os.environ.get("MARBLE_EVAL_MODEL")
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
    if spec.single_agent:
        if not cfg["agents"]:
            raise ValueError(f"task {task.task_id} has no agents for single_agent baseline")
        chosen = cfg["agents"][0]
        chosen_id = chosen.get("agent_id")
        cfg["agents"] = [chosen]
        # A single-agent episode has no valid cross-agent targets.
        cfg["relationships"] = []
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
    if spec.uses_memory:
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
        expanded += list(BASELINES) if b == "multi" else [canonical_baseline(b)]
    seen: List[str] = []
    for b in expanded:
        if b not in seen:
            seen.append(b)
    return [(b, t) for b in seen for t in tasks]


def _new_run_root(base_root: str | Path, descriptor: str) -> Path:
    base = Path(base_root)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.localtime())
    while True:
        candidate = base / f"{descriptor}_{timestamp}_{secrets.token_hex(3)}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate


def _append_run_event(
    run_root: str | Path,
    event: str,
    task: BenchmarkTask,
    **details: Any,
) -> None:
    record = {
        "event": event,
        "timestamp": time.time(),
        "baseline": canonical_baseline(details.pop("baseline", "")),
        "benchmark": task.benchmark,
        "task_id": task.task_id,
        **details,
    }
    path = Path(run_root) / "run_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_task(
    task: BenchmarkTask,
    baseline: str,
    out_root: str | Path,
    *,
    dry_run: bool = False,
    seed: Optional[int] = None,
    max_iterations: Optional[int] = None,
    max_cards: int = 5,
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
    lambda_: float = 0.05,
    beta: float = 0.25,
    manifest: Optional[str] = None,
    provider: Optional[str] = None,
    task_timeout: Optional[float] = 900.0,
    enable_comm_governor: bool = False,
) -> Dict[str, Any]:
    if lambda_ < 0:
        raise ValueError("lambda must be non-negative")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    baseline = canonical_baseline(baseline)
    if baseline == "ours_private_to_global" and not ablation:
        ablation = "visibility:private_to_global"
    if provider:
        _configure_provider(provider)
    if seed is not None:
        random.seed(seed)

    # Ensure invalid IPv6 NO_PROXY like :1 doesn't crash httpx
    for var in ("NO_PROXY", "no_proxy"):
        val = os.environ.get(var, "")
        if ":1" in val or "::1" in val:
            os.environ.pop(var, None)

    out_root = Path(out_root).resolve()
    tdir = (out_root / baseline / task.benchmark / str(task.task_id)).resolve()
    tdir.mkdir(parents=True, exist_ok=True)

    # Isolate database fixture writes into task output dir
    if task.benchmark == "database":
        os.environ["MARBLE_DATASET_LOG"] = str((tdir / "dataset.txt").resolve())
        os.environ["MARBLE_BADSQL_LOG"] = str((tdir / "badsql.txt").resolve())

    _append_run_event(out_root, "task_start", task, baseline=baseline)

    errors: List[str] = []

    cfg = task_config(
        task,
        baseline,
        max_cards,
        max_reads_per_step,
        retriever,
        llm,
        ablation,
        provider,
    )
    # ponytail: ablations mutate the config memory block (policy->controller name,
    # retrieval:none->max_cards=0). make_controller is keyed by baseline, so the
    # runner must read the *effective* controller/max_cards back out of config.
    effective_baseline = baseline
    effective_max_cards = max_cards
    spec = baseline_spec(baseline)
    effective_comm_gov = enable_comm_governor or spec.enable_comm_governor or (
        os.environ.get("MARBLE_ENABLE_COMM_GOVERNOR", "").lower() in ("1", "true")
    )
    if ablation and ("comm_gov:off" in ablation or "no_comm_gov" in ablation):
        effective_comm_gov = False
    mem_block = cfg.get("memory")
    if spec.uses_memory and isinstance(mem_block, dict):
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
        "agent_count": len(task.agents),
        "seed": seed,
        "ablation": ablation,
        "manifest": manifest,
        "provider": provider,
        "comm_governor": effective_comm_gov,
        "status": "ok",
        # spec §11.2/§11.3: record the exact models used, all non-empty
        "worker_model": cfg["llm"],
        "controller_model": describe_controller(
            baseline,
            controller_checkpoint=controller_checkpoint,
            qwen_base_model=qwen_base_model,
            qwen_api_model=qwen_api_model,
            ablation=ablation,
        ),
        "evaluator_model": cfg["metrics"].get("evaluate_llm", ""),
        "retrieval": {
            "name": retriever,
            "max_cards": effective_max_cards,
            "max_reads_per_step": max_reads_per_step,
        },
        "reward_config": {"lambda": lambda_, "beta": beta},
    }
    summary["setting"] = "|".join(
        (
            f"method={baseline}",
            f"ablation={ablation or 'none'}",
            f"comm_gov={'on' if effective_comm_gov else 'off'}",
            f"retrieval={retriever}",
            f"max_cards={effective_max_cards}",
            f"max_reads_per_step={max_reads_per_step}",
            f"lambda={lambda_:g}",
            f"beta={beta:g}",
        )
    )
    completed = tdir / "summary.json"
    if completed.exists():
        try:
            previous = json.loads(completed.read_text(encoding="utf-8"))
            status_ok = previous.get("status") in {"ok", "score_unavailable"}
            if "setting" in previous:
                setting_ok = previous["setting"] == summary["setting"]
            else:
                setting_ok = previous.get("method") == summary["method"]
            seed_ok = (previous["seed"] == seed) if ("seed" in previous and seed is not None) else True
            worker_model_ok = (previous["worker_model"] == summary["worker_model"]) if "worker_model" in previous else True
            controller_model_ok = (previous["controller_model"] == summary["controller_model"]) if "controller_model" in previous else True
            provider_ok = (previous["provider"] == summary["provider"]) if "provider" in previous else True

            if status_ok and setting_ok and seed_ok and worker_model_ok and controller_model_ok and provider_ok:
                _append_run_event(
                    out_root,
                    "task_skip",
                    task,
                    baseline=baseline,
                    status=previous["status"],
                )
                return previous
            else:
                mismatches = []
                if not status_ok:
                    mismatches.append(f"status={previous.get('status')}")
                if not setting_ok:
                    mismatches.append(f"setting: previous='{previous.get('setting')}' vs expected='{summary['setting']}'")
                if not seed_ok:
                    mismatches.append(f"seed: previous={previous.get('seed')} vs expected={seed}")
                if not worker_model_ok:
                    mismatches.append(f"worker_model: previous='{previous.get('worker_model')}' vs expected='{summary['worker_model']}'")
                if not controller_model_ok:
                    mismatches.append(f"controller_model: previous='{previous.get('controller_model')}' vs expected='{summary['controller_model']}'")
                if not provider_ok:
                    mismatches.append(f"provider: previous='{previous.get('provider')}' vs expected='{summary['provider']}'")
                print(f"[info] Task {task.benchmark}:{task.task_id} existing summary config mismatch ({'; '.join(mismatches)}). Re-running...")
        except Exception as exc:
            print(f"[warn] Failed to read existing summary.json for {task.benchmark}:{task.task_id}: {exc}")

    if dry_run:
        summary["status"] = "dry_run"
        (tdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _append_run_event(
            out_root, "task_end", task, baseline=baseline, status=summary["status"]
        )
        return summary

    old_handler = None
    if task_timeout and task_timeout > 0 and hasattr(signal, "SIGALRM"):
        def _alarm_handler(signum, frame):
            raise TaskTimeoutError(
                f"Task execution exceeded timeout limit of {task_timeout}s"
            )
        try:
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(int(task_timeout))
        except (ValueError, AttributeError):
            old_handler = None

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
            lambda_=lambda_,
            beta=beta,
            provider=provider,
            enable_comm_governor=effective_comm_gov,
        )
        summary.update(metrics)
        if metrics.get("score_status") != "available":
            summary["status"] = "score_unavailable"
    except TaskTimeoutError as exc:
        summary["status"] = "timeout"
        errors.append(f"TaskTimeoutError: {exc}")
        errors.append(traceback.format_exc())
        _append_run_event(
            out_root,
            "task_timeout",
            task,
            baseline=baseline,
            timeout=task_timeout,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 — one bad task must not kill the sweep
        summary["status"] = "error"
        errors.append(f"{type(exc).__name__}: {exc}")
        errors.append(traceback.format_exc())
        _append_run_event(
            out_root,
            "task_error",
            task,
            baseline=baseline,
            error_type=type(exc).__name__,
            error=str(exc),
        )
    finally:
        if old_handler is not None and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    if errors:
        (tdir / "errors.log").write_text("\n".join(errors) + "\n", encoding="utf-8")
    (tdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _append_run_event(
        out_root, "task_end", task, baseline=baseline, status=summary["status"]
    )
    return summary


def _require_worker_key(provider: str = "sensenova") -> None:
    import os

    key_vars = _PROVIDER_DEFAULTS.get(provider, {}).get("key_vars", _WORKER_KEY_VARS)
    if not any(os.environ.get(k) for k in key_vars):
        raise RuntimeError(
            "real rollout needs a worker API key: set one of "
            f"{', '.join(key_vars)} (or use --dry-run)"
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
    lambda_: float = 0.05,
    beta: float = 0.25,
    provider: str = "sensenova",
    enable_comm_governor: bool = False,
) -> Dict[str, Any]:
    _require_worker_key(provider)
    import os

    from marble.configs.config import Config
    from marble.experiments.evaluate import evaluate_memory_trace
    from marble.llms import ApiUsageMeter
    from marble.memory.rewards import proposal_rewards

    started_at = time.monotonic()
    mem = make_memory_runtime(
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
        worker_model=cfg["llm"],
    )
    # ponytail: selector "top" makes agents read the top-ranked cards without an
    # extra API call, so memory actually influences the task
    if retriever not in ("key_first", "visible_k", "none"):
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
    comm_gov = None
    if enable_comm_governor or os.environ.get("MARBLE_ENABLE_COMM_GOVERNOR", "").lower() in ("1", "true"):
        from marble.engine.communication_governor import CommunicationGovernor
        comm_gov = CommunicationGovernor()
    harness = MemoryStep(
        mem, max_cards=eff_max_cards, max_reads_per_step=max_reads_per_step,
        selector="top", task_goal=task.task, agent_role_map=agent_role_map,
        baseline=baseline, worker_model=cfg["llm"], comm_governor=comm_gov,
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
    usage_meter = ApiUsageMeter()
    os.chdir(marble_dir)
    engine = None
    try:
        with usage_meter:
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
        if engine is not None:
            env = getattr(engine, "environment", None)
            if env is not None and hasattr(env, "terminate"):
                try:
                    env.terminate()
                except Exception as exc:
                    print(f"[warn] env.terminate failed during episode cleanup: {exc}")

    api_usage = usage_meter.snapshot()
    metrics: Dict[str, Any] = {
        "episode_latency_s": round(time.monotonic() - started_at, 6),
        "worker_tokens": sum(
            int(agent.get_token_usage())
            for agent in getattr(engine, "agents", [])
            if hasattr(agent, "get_token_usage")
        ),
        "api_calls": api_usage["api_calls"],
        "input_tokens": api_usage["input_tokens"],
        "output_tokens": api_usage["output_tokens"],
        "total_tokens": api_usage["input_tokens"] + api_usage["output_tokens"],
    }
    evaluator = getattr(engine, "evaluator", None)
    evaluator_metrics = getattr(evaluator, "metrics", None)
    if isinstance(evaluator_metrics, dict):
        result_text = ""
        out_path = Path(cfg["output"].get("file_path", ""))
        if out_path.exists():
            result_text = out_path.read_text(encoding="utf-8", errors="ignore")
        score = _score_result(
            task.benchmark,
            evaluator,
            task.task,
            result_text,
            getattr(engine, "environment", None),
        )
        metrics.update(score)
        metrics["engine_metrics"] = {
            k: v for k, v in evaluator_metrics.items() if isinstance(v, (int, float, str))
        }
    else:
        metrics["score_status"] = "unavailable"

    events = _read_jsonl(tdir / "memory_trace.jsonl")
    memory_metrics = evaluate_memory_trace(
        events,
        task_success=metrics.get("task_success"),
        task_score=float(metrics.get("task_score") or 0.0),
    )
    metrics["memory_metrics"] = memory_metrics
    for metric_name in (
        "active_memory_count",
        "private_written",
        "private_read",
        "private_owner_reuse",
        "global_non_owner_reuse",
        "cross_agent_exposure",
        "cross_agent_reads",
        "negative_transfer",
    ):
        if metric_name in memory_metrics:
            metrics[metric_name] = memory_metrics[metric_name]
    metrics["memory_cost"] = measured_memory_cost(events)
    metrics["controller_api_calls"] = sum(
        int(event.get("manager_api_calls", 0) or 0)
        + int(event.get("api_calls", 0) or 0)
        for event in events
        if event.get("event") in {"memory_r1_operation", "memory_r1_distillation"}
    )
    metrics["controller_tokens"] = sum(
        int(event.get("manager_input_tokens", 0) or 0)
        + int(event.get("manager_output_tokens", 0) or 0)
        + int(event.get("input_tokens", 0) or 0)
        + int(event.get("output_tokens", 0) or 0)
        for event in events
        if event.get("event") in {"memory_r1_operation", "memory_r1_distillation"}
    )
    metrics["episode_reward"] = float(metrics.get("task_score") or 0.0) - lambda_ * metrics["memory_cost"]
    reward_kw: Dict[str, float] = {}
    if ablation:
        factor, option = parse_ablation(ablation)
        reward_kw = reward_override(factor, option)
    effective_reward_kw = {"beta": beta, "lambda_": lambda_}
    effective_reward_kw.update(reward_kw)
    credits = proposal_rewards(
        events,
        task_score=float(metrics.get("task_score") or 0.0),
        same_task_baseline=0.0,
        task_success=metrics.get("task_success"),
        **effective_reward_kw,
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
def _score_result(
    benchmark: str,
    evaluator,
    task_content: str,
    result_text: str,
    environment: Any,
) -> Dict[str, Any]:
    """Return only benchmark-specific, normalized and inspectable task scores."""
    metrics = getattr(evaluator, "metrics", {}) or {}
    try:
        if benchmark == "research":
            values = _valid_ratings((metrics.get("task_evaluation") or {}).values())
            if not values and result_text and evaluator is not None and hasattr(evaluator, "evaluate_task_research"):
                evaluator.evaluate_task_research(task_content, result_text)
                values = _valid_ratings((metrics.get("task_evaluation") or {}).values())
            return _rating_result(values)
        if benchmark == "coding":
            values = _valid_ratings((metrics.get("code_quality") or {}).values())
            if not values and result_text and evaluator is not None and hasattr(evaluator, "evaluate_code_quality"):
                evaluator.evaluate_code_quality(task_content, result_text)
                values = _valid_ratings((metrics.get("code_quality") or {}).values())
            result = _rating_result(values)
            if values:
                result["task_success"] = (
                    (metrics.get("code_quality", {}).get("executability", 0) >= 4)
                    and (metrics.get("code_quality", {}).get("instruction_following", 0) >= 4)
                )
            return result
        if benchmark == "database":
            evaluation = metrics.get("task_evaluation") or {}
            targets = {
                str(value).upper()
                for value in evaluation.get("root_cause", [])
                if isinstance(value, str)
            }
            predicted_text = str(evaluation.get("predicted", result_text) or "")
            if not targets or not predicted_text.strip():
                return {"score_status": "unavailable", "task_score": None, "task_success": None}
            predicted = {
                label
                for label in targets
                if re.search(rf"\b{re.escape(label)}\b", predicted_text.upper())
            }
            overlap = len(targets & predicted)
            score = 2 * overlap / (len(targets) + len(predicted)) if predicted else 0.0
            return {
                "score_status": "available",
                "task_score": score,
                "task_success": predicted == targets,
                "score_detail": {"target_causes": sorted(targets), "predicted_causes": sorted(predicted)},
            }
        if benchmark == "bargaining":
            evaluator.evaluate_task_world(task_content, result_text)
            rating_values = _valid_ratings(
                value
                for side in (metrics.get("task_evaluation") or {}).values()
                if isinstance(side, dict)
                for value in side.values()
            )
            result = _rating_result(rating_values)
            if result["score_status"] == "available":
                result["task_success"] = bool(
                    getattr(environment, "agreement_reached", False)
                )
            return result
        if benchmark == "minecraft":
            score = metrics.get("task_evaluation")
            if isinstance(score, (int, float)) and 0 <= score <= 5:
                return {
                    "score_status": "available",
                    "task_score": score / 5.0,
                    "task_success": score > 0,
                }
    except Exception as exc:  # noqa: BLE001 — eval LLM may be absent offline
        print(f"[warn] benchmark scoring failed ({benchmark}): {exc}")
        return {"score_status": "error", "task_score": None, "task_success": None}
    return {"score_status": "unavailable", "task_score": None, "task_success": None}


def _valid_ratings(values) -> List[float]:
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and 1 <= float(value) <= 5
    ]


def _rating_result(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"score_status": "unavailable", "task_score": None, "task_success": None}
    score = sum(values) / len(values) / 5.0
    return {
        "score_status": "available",
        "task_score": score,
        "task_success": all(value >= 4 for value in values),
    }


def _benchmark_score(benchmark: str, evaluator, task_content: str, result_text: str) -> float:
    """Legacy float helper retained for callers that do not consume score status."""
    return float(
        _score_result(benchmark, evaluator, task_content, result_text, None).get(
            "task_score"
        )
        or 0.0
    )


# ------------------------------------------------------------------------ CLI
def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run governed-memory baselines on MultiAgentBench")
    ap.add_argument("--benchmark", required=True,
                    help=f"one of {BENCHMARKS}, comma-separated, or 'all'")
    ap.add_argument("--baseline", default="heuristic",
                    help=f"one of {BASELINES} or 'multi'")
    ap.add_argument("--split", default="all", help="all|train|test (deterministic by task_id)")
    ap.add_argument("--manifest", default=None, help="frozen task manifest JSON")
    ap.add_argument("--task-ids", default="", help="comma-separated task ids")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--retrieval", default="key_first")
    ap.add_argument("--max-cards", type=int, default=5)
    ap.add_argument("--max-reads-per-step", type=int, default=2)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=0.15)
    ap.add_argument("--beta", type=float, default=0.25)
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
    ap.add_argument("--provider", choices=tuple(_PROVIDER_DEFAULTS), default=None)
    ap.add_argument("--ablation", default=None, help="factor:option, e.g. retrieval:none")
    ap.add_argument(
        "--task-timeout",
        type=float,
        default=float(os.environ.get("MARBLE_TASK_TIMEOUT", "900.0")),
        help="max execution time in seconds per task before raising timeout (default: 900.0)",
    )
    ap.add_argument(
        "--enable-comm-governor",
        action="store_true",
        default=os.environ.get("MARBLE_ENABLE_COMM_GOVERNOR", "").lower() in ("1", "true"),
        help="enable CommunicationGovernor to compress echo-restatements between agents",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--out",
        default=None,
        help="explicit run root; omit to create a timestamped run under runs/",
    )
    return ap


def main(argv: Optional[List[str]] = None) -> None:
    ap = _build_arg_parser()
    args = ap.parse_args(argv)
    if args.provider:
        _configure_provider(args.provider)

    # ponytail: global floor so any untimeouted raw call (arxiv fetch, requests) can't hang forever
    socket.setdefaulttimeout(int(os.environ.get("MARBLE_SOCKET_TIMEOUT", "120")))

    task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()] or None
    bench_names = [b.strip() for b in args.benchmark.split(",") if b.strip()]
    if "all" in bench_names:
        bench_names = list(BENCHMARKS)
    effective_seed = args.seed
    if effective_seed is None and args.manifest:
        manifest_seed = read_manifest(args.manifest).get("seed")
        if manifest_seed is not None:
            if not isinstance(manifest_seed, int) or isinstance(manifest_seed, bool):
                raise ValueError("manifest seed must be an integer")
            effective_seed = manifest_seed
    tasks: List[BenchmarkTask] = []
    for bn in bench_names:
        if bn not in BENCHMARKS:
            raise ValueError(f"unknown benchmark {bn!r}; choose from {BENCHMARKS} or 'all'")
    if args.manifest:
        tasks = [
            task for task in load_manifest_tasks(args.manifest, split=args.split)
            if task.benchmark in bench_names
        ]
        if task_ids is not None:
            wanted = set(task_ids)
            tasks = [task for task in tasks if task.task_id in wanted]
    else:
        for bn in bench_names:
            tasks += load_tasks(bn, start=args.start, task_ids=task_ids)
        tasks = _apply_split(tasks, args.split)
    if args.limit:
        tasks = tasks[: args.limit]
    # --baseline multi expands to the frozen main-method matrix.
    baselines = ["multi"] if args.baseline == "multi" else [args.baseline]
    runs = plan_runs(baselines, tasks)

    run_id = f"{args.benchmark}_{args.baseline}"
    if args.ablation:
        run_id += f"_ablation-{args.ablation.replace(':', '_')}"
    if args.enable_comm_governor:
        run_id += "_commgov"
    if effective_seed is not None:
        run_id += f"_seed{effective_seed}"
    run_id += f"_cards{args.max_cards}_reads{args.max_reads_per_step}"
    run_id += f"_lambda{args.lambda_:g}_beta{args.beta:g}"
    out_root = (
        Path(args.out)
        if args.out is not None
        else _new_run_root("runs", run_id)
    )
    print(f"{len(runs)} episode(s) -> {out_root}")
    for baseline, task in runs:
        summary = run_task(
            task, baseline, out_root,
            dry_run=args.dry_run,
            seed=effective_seed,
            max_iterations=args.max_iterations,
            max_cards=args.max_cards,
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
            lambda_=args.lambda_,
            beta=args.beta,
            manifest=args.manifest,
            provider=args.provider,
            task_timeout=args.task_timeout,
            enable_comm_governor=args.enable_comm_governor,
        )
        print(f"  [{summary['status']}] {baseline} task={task.task_id}")


if __name__ == "__main__":
    main()
