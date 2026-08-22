"""Aggregate baseline rollout outputs into comparison metrics."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def token_count(text: str) -> int:
    return len(text.split())


def evaluate_baseline(summary_entry: List[Dict[str, Any]], trace_path: str) -> Dict[str, Any]:
    events = _read_jsonl(trace_path)
    visibility_counts: Dict[str, int] = {"private": 0, "global": 0, "absent": 0}
    global_tokens = 0
    read_count = 0
    cross_agent_reads = 0
    owners: Dict[str, str] = {}
    for ev in events:
        kind = ev.get("event")
        if kind == "memory_decision" and ev.get("memory_id"):
            vis = ev["target"].get("visibility")
            visibility_counts[vis] = visibility_counts.get(vis, 0) + 1
            owners[ev["memory_id"]] = ev["proposal"]["agent_id"]
            if vis == "global":
                global_tokens += token_count(ev["proposal"].get("raw_value", ""))
        elif kind == "memory_read":
            read_count += 1
            if owners.get(ev["memory_id"]) != ev["reader_id"]:
                cross_agent_reads += 1

    episodes = len(summary_entry)
    return {
        "episodes": episodes,
        "compile_rate": (
            sum(1 for r in summary_entry if r["compile_ok"]) / episodes if episodes else 0.0
        ),
        "proposals_stored": sum(r["proposals_stored"] for r in summary_entry),
        "visible_key_events": sum(r["visible_key_events"] for r in summary_entry),
        "decisions": visibility_counts,
        "global_tokens": global_tokens,
        "reads": read_count,
        "cross_agent_reads": cross_agent_reads,
    }


def evaluate_run(run_dir: str) -> Dict[str, Any]:
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    report: Dict[str, Any] = {}
    for baseline, entry in summary.items():
        trace_path = os.path.join(run_dir, baseline, "trace.jsonl")
        report[baseline] = evaluate_baseline(entry, trace_path)
    return report


def format_report(report: Dict[str, Any]) -> str:
    cols = ["episodes", "compile_rate", "proposals_stored", "visible_key_events",
            "reads", "cross_agent_reads", "global_tokens"]
    lines = ["baseline".ljust(16)] + [c.rjust(18) for c in cols]
    for baseline, m in report.items():
        row = [baseline.ljust(16)]
        for c in cols:
            val = m[c] if c != "decisions" else sum(m["decisions"].values())
            row.append(str(val).rjust(18))
        lines.append("".join(row))
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate governed-memory baselines.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(format_report(evaluate_run(args.run_dir)))
