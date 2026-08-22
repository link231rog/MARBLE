"""Real-LLM rollout over a minimal MARBLE-style coding flow with governed memory."""

from __future__ import annotations

import json
import os
import py_compile
import re
import time
import urllib.request
from typing import Any, Dict, List, Optional

from marble.memory.adapter import MemoryAwareAgentAdapter, format_cards

CODER_SYSTEM = (
    "You are a senior Python developer. Solve the given task. "
    "Reply with exactly one python code block containing solution.py."
)
REVIEWER_SYSTEM = (
    "You are a code reviewer. Revise the previous solution if shared notes are available, "
    "otherwise write your own solution from scratch. Reply with exactly one python code block."
)


def _chat(base_url: str, model: str, key: str, system: str, user: str,
          max_tokens: int, timeout: float) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"] or ""
        except Exception as err:  # noqa: BLE001 - retry any transport failure
            last_err = err
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after retries: {last_err}")


class NvidiaLLM:
    """Thin OpenAI-compatible client for NVIDIA NIM endpoints (env-configured)."""

    def __init__(self) -> None:
        self.base_url = os.environ.get("NVAPI_BASE", "https://integrate.api.nvidia.com/v1")
        self.model = os.environ.get("NVAPI_MODEL", "deepseek-ai/deepseek-v4-flash-0731")
        key = os.environ.get("NVAPI_KEY")
        if not key:
            raise RuntimeError("set NVAPI_KEY in the environment")
        self.key = key
        self.max_tokens = int(os.environ.get("NVAPI_MAX_TOKENS", "1024"))
        self.timeout = float(os.environ.get("NVAPI_TIMEOUT", "300"))

    def act(self, system: str, user: str) -> str:
        return _chat(self.base_url, self.model, self.key, system, user,
                     self.max_tokens, self.timeout)


_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = _FENCE.findall(text)
    return blocks[-1].strip() if blocks else text.strip()


def run_episode(
    task_id: str,
    task: str,
    llm: Any,
    memory: Any,
    workspace_dir: str,
) -> Dict[str, Any]:
    """One coder->reviewer episode; returns governance + outcome stats."""
    os.makedirs(workspace_dir, exist_ok=True)
    path = os.path.join(workspace_dir, "solution.py")

    def memories_block(cards: List[Any]) -> str:
        return format_cards(cards)

    coder_adapter = MemoryAwareAgentAdapter(llm, memory, source="worker") if memory else None
    reviewer_adapter = MemoryAwareAgentAdapter(llm, memory, source="worker") if memory else None

    reads = 0
    visible_key_events = 0
    stored: List[str] = []

    # coder turn
    if coder_adapter is not None:
        cards = coder_adapter.before_step(task_id, "coder", query=task)
        visible_key_events += bool(cards)
        mem_section = (
            "Notes from teammates:\n" + memories_block(cards)
            if cards else ""
        )
    else:
        mem_section = ""
    coder_out = llm.act(CODER_SYSTEM, task + ("\n\n" + mem_section if mem_section else ""))
    code = extract_code(coder_out)
    if coder_adapter is not None:
        mid = coder_adapter.after_step(coder_out)
        if mid:
            stored.append(mid)

    # reviewer turn
    review_context = f"Previous solution:\n```python\n{code}\n```\n"
    if reviewer_adapter is not None:
        cards = reviewer_adapter.before_step(task_id, "reviewer", query=task)
        visible_key_events += bool(cards)
        sections = []
        for card in cards:
            try:
                sections.append(reviewer_adapter.read(card.memory_id))
                reads += 1
            except (KeyError, PermissionError):
                continue
        if sections:
            review_context += "\nShared notes:\n" + "\n---\n".join(sections)
    reviewer_out = llm.act(REVIEWER_SYSTEM, task + "\n\n" + review_context)
    revised = extract_code(reviewer_out)
    final_code = revised or code
    if reviewer_adapter is not None:
        mid = reviewer_adapter.after_step(reviewer_out)
        if mid:
            stored.append(mid)

    with open(path, "w") as fh:
        fh.write(final_code + "\n")
    compile_ok = True
    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError:
        compile_ok = False

    return {
        "task_id": task_id,
        "proposals_stored": len(stored),
        "visible_key_events": visible_key_events,
        "reads": reads,
        "compile_ok": compile_ok,
        "solution_path": path,
    }


DEFAULT_TASKS = {
    "fizzbuzz": "Write solution.py that prints numbers 1..30, replacing multiples of 3 with Fizz, of 5 with Buzz, of both with FizzBuzz.",
    "wordcount": "Write solution.py defining count_words(text: str) -> dict mapping lowercase words to counts, splitting on non-alphanumeric characters.",
}


def run_baselines(
    baselines: List[str],
    out_dir: str,
    llm: Any,
    tasks: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    from marble.experiments.baselines import BASELINES, make_memory

    tasks = tasks or DEFAULT_TASKS
    unknown = [b for b in baselines if b not in BASELINES]
    if unknown:
        raise ValueError(f"unknown baselines: {unknown}")

    summary: Dict[str, Any] = {}
    for baseline in baselines:
        base_dir = os.path.join(out_dir, baseline)
        os.makedirs(base_dir, exist_ok=True)
        trace_path = os.path.join(base_dir, "trace.jsonl")
        results = []
        for task_id, task in tasks.items():
            memory = make_memory(baseline, trace_path=trace_path)
            ws = os.path.join(base_dir, task_id, "workspace")
            results.append(run_episode(task_id, task, llm, memory, ws))
        summary[baseline] = results
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run governed-memory coding baselines.")
    parser.add_argument("--baselines", nargs="+",
                        default=["no_memory", "global_always", "private_only"])
    parser.add_argument("--out", default="runs/coding_baselines")
    args = parser.parse_args()
    result = run_baselines(args.baselines, args.out, NvidiaLLM())
    print(json.dumps(result, indent=2))
