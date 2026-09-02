"""Aggregate baseline rollout outputs into comparison metrics."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from marble.experiments.task_manifest import read_manifest


def setting_key(summary: Dict[str, Any]) -> str:
    """Return the explicit experiment setting, with legacy method fallback."""
    explicit = summary.get("setting")
    if explicit:
        return str(explicit)
    method = summary.get("method")
    if not any(key in summary for key in ("ablation", "retrieval", "reward_config")):
        return str(method)
    retrieval = summary.get("retrieval") or {}
    reward = summary.get("reward_config") or {}
    return "|".join(
        (
            f"method={method}",
            f"ablation={summary.get('ablation') or 'none'}",
            f"retrieval={retrieval.get('name', 'unknown')}",
            f"max_cards={retrieval.get('max_cards', 'unknown')}",
            f"max_reads_per_step={retrieval.get('max_reads_per_step', 'unknown')}",
            f"lambda={reward.get('lambda', 'unknown')}",
            f"beta={reward.get('beta', 'unknown')}",
        )
    )


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
            row.append(str(m[c]).rjust(18))
        lines.append("".join(row))
    return "\n".join(lines)


# ------------------------------------------------------- benchmark-run metrics
def evaluate_memory_trace(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Memory metrics from one memory_trace.jsonl (spec §13)."""
    decisions = [e for e in events if e.get("event") == "memory_decision"]
    r1_operations = [
        e for e in events if e.get("event") == "memory_r1_operation"
    ]
    stored = [e for e in decisions if e.get("memory_id")]
    total = len(stored)
    by_vis: Dict[str, int] = {"private": 0, "global": 0, "absent": 0}
    owners: Dict[str, str] = {}
    superseded_ids = set()
    active_items: Dict[str, Dict[str, Any]] = {}
    for e in stored:
        vis = e["target"].get("visibility")
        by_vis[vis] = by_vis.get(vis, 0) + 1
        owners[e["memory_id"]] = e["proposal"]["agent_id"]
        if e["target"].get("supersedes"):
            superseded_ids.add(e["target"]["supersedes"])
        memory_id = e["memory_id"]
        active_items[memory_id] = {
            "visibility": vis,
            "tokens": token_count(e["proposal"].get("raw_value", "")),
        }
        if e["target"].get("supersedes"):
            active_items.pop(e["target"]["supersedes"], None)
    for e in r1_operations:
        operation = e.get("operation")
        memory_id = e.get("memory_id")
        if not isinstance(memory_id, str):
            continue
        if operation in {"ADD", "UPDATE"}:
            proposal = e.get("proposal") or {}
            if isinstance(proposal.get("agent_id"), str):
                owners[memory_id] = proposal["agent_id"]
            active_items[memory_id] = {
                "visibility": "global",
                "tokens": token_count(proposal.get("raw_value", "")),
            }
        elif operation == "DELETE":
            active_items.pop(memory_id, None)
    reads = [
        e for e in events
        if e.get("event") == "memory_read" and isinstance(e.get("memory_id"), str)
    ]
    exposures = [e for e in events if e.get("event") == "memory_exposure"]
    cross = sum(1 for e in reads if owners.get(e["memory_id"]) != e.get("reader_id"))
    read_ids = {e["memory_id"] for e in reads if e.get("memory_id")}
    reused = sum(1 for mid in read_ids if mid in owners)
    active_global = [
        item for item in active_items.values() if item["visibility"] == "global"
    ]
    active_private = [
        item for item in active_items.values() if item["visibility"] == "private"
    ]
    return {
        "decisions_total": total,
        "accept_rate": (total / len(decisions)) if decisions else 0.0,
        "reject_rate": (
            sum(1 for e in decisions if not e.get("memory_id")) / len(decisions)
        ) if decisions else 0.0,
        "private": by_vis.get("private", 0),
        "global": by_vis.get("global", 0),
        "supersessions": len(superseded_ids),
        "r1_adds": sum(1 for e in r1_operations if e.get("operation") == "ADD"),
        "r1_updates": sum(1 for e in r1_operations if e.get("operation") == "UPDATE"),
        "r1_deletes": sum(1 for e in r1_operations if e.get("operation") == "DELETE"),
        "r1_noops": sum(1 for e in r1_operations if e.get("operation") == "NOOP"),
        "exposure_events": len(exposures),
        "exposed_cards": sum(
            len(e.get("memory_ids", []))
            for e in exposures
            if isinstance(e.get("memory_ids"), list)
        ),
        "reads": len(reads),
        "cross_agent_reads": cross,
        "reuse_rate": (reused / total) if total else 0.0,
        "active_memory_count": len(active_items),
        "active_global_count": len(active_global),
        "active_private_count": len(active_private),
        "active_memory_tokens": sum(item["tokens"] for item in active_items.values()),
        "active_global_tokens": sum(item["tokens"] for item in active_global),
        "active_private_tokens": sum(item["tokens"] for item in active_private),
    }


def evaluate_task_dir(task_dir: str | os.PathLike) -> Dict[str, Any]:
    """Join summary.json + memory_trace.jsonl + reward.json for one task."""
    task_dir = os.fspath(task_dir)
    with open(os.path.join(task_dir, "summary.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    row: Dict[str, Any] = {
        "method": summary.get("method"),
        "setting": setting_key(summary),
        "benchmark": summary.get("benchmark"),
        "task_id": summary.get("task_id"),
        "agent_count": summary.get("agent_count"),
        "seed": summary.get("seed"),
        "ablation": summary.get("ablation"),
        "manifest": summary.get("manifest"),
        "status": summary.get("status"),
        "task_score": summary.get("task_score", 0.0),
        "task_success": summary.get("task_success"),
        "score_status": summary.get("score_status", "unavailable"),
        "episode_latency_s": summary.get("episode_latency_s"),
        "worker_tokens": summary.get("worker_tokens"),
        "api_calls": summary.get("api_calls"),
        "total_tokens": summary.get("total_tokens"),
        "controller_api_calls": summary.get("controller_api_calls"),
        "controller_tokens": summary.get("controller_tokens"),
        "memory_cost": summary.get("memory_cost"),
        "episode_reward": summary.get("episode_reward"),
        "retrieval": summary.get("retrieval", {}),
        "reward_config": summary.get("reward_config", {}),
        "memory": evaluate_memory_trace(
            _read_jsonl(os.path.join(task_dir, "memory_trace.jsonl"))
        ),
    }
    reward_path = os.path.join(task_dir, "reward.json")
    if os.path.exists(reward_path):
        with open(reward_path, encoding="utf-8") as fh:
            credits = json.load(fh)
        row["reward"] = sum(credits.values())
    return row


def evaluate_run_root(
    run_root: str | os.PathLike,
    *,
    include_unavailable: bool = False,
    manifest: str | os.PathLike | None = None,
    split: str = "all",
) -> List[Dict[str, Any]]:
    """Task rows under runs/<run_id>/<benchmark>/<task_id>/."""
    if split not in {"all", "train", "test"}:
        raise ValueError("manifest split must be all|train|test")
    allowed = None
    if manifest is not None:
        payload = read_manifest(manifest)
        if "splits" in payload:
            records = (
                list(payload["splits"].get("train", []))
                + list(payload["splits"].get("test", []))
                if split == "all"
                else list(payload["splits"].get(split, []))
            )
        else:
            records = list(payload.get("tasks", []))
        allowed = {
            (str(record["benchmark"]), int(record["task_id"]))
            for record in records
        }
    root = os.fspath(run_root)
    rows: List[Dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "summary.json" in filenames:
            row = evaluate_task_dir(dirpath)
            if allowed is not None and (
                str(row.get("benchmark")), int(row.get("task_id"))
            ) not in allowed:
                continue
            if include_unavailable or row.get("score_status") == "available":
                rows.append(row)
    return rows


def aggregate_by_method(
    rows: List[Dict[str, Any]],
    *,
    include_unavailable: bool = False,
) -> Dict[str, Dict[str, float]]:
    """mean ± standard error per method over numeric leaf metrics."""
    import math

    def leaves(d: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(leaves(v, f"{prefix}{k}."))
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                out[f"{prefix}{k}"] = float(v)
        return out

    grouped: Dict[str, List[Dict[str, float]]] = {}
    for r in rows:
        if not include_unavailable and r.get("score_status") != "available":
            continue
        grouped.setdefault(str(r.get("setting") or r["method"]), []).append(leaves(r))
    report: Dict[str, Dict[str, float]] = {}
    for method, metric_dicts in sorted(grouped.items()):
        keys = set().union(*(d.keys() for d in metric_dicts))
        agg: Dict[str, float] = {}
        for k in sorted(keys):
            vals = [d[k] for d in metric_dicts if k in d]
            mean = sum(vals) / len(vals)
            se = (
                math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals) / len(vals))
                if len(vals) > 1
                else 0.0
            )
            agg[k] = round(mean, 6)
            agg[k + "_se"] = round(se, 6)
        report[method] = agg
    return report


def main(argv: List[str] | None = None) -> None:
    import argparse
    import glob as _glob

    parser = argparse.ArgumentParser(description="Evaluate governed-memory baselines.")
    parser.add_argument("--run-dir", required=True, help="coding_rollout run dir OR run_root")
    parser.add_argument(
        "--include-unavailable",
        action="store_true",
        help="include rows whose score is unavailable for diagnostics",
    )
    parser.add_argument("--manifest", help="frozen task manifest used for filtering")
    parser.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="manifest split to evaluate",
    )
    args = parser.parse_args(argv)

    # any summary.json below the root that is not the legacy root-level one
    legacy = os.path.abspath(os.path.join(args.run_dir, "summary.json"))
    nested = [
        p for p in _glob.glob(os.path.join(args.run_dir, "**", "summary.json"), recursive=True)
        if os.path.abspath(p) != legacy
    ]
    if nested:
        rows = evaluate_run_root(
            args.run_dir,
            include_unavailable=args.include_unavailable,
            manifest=args.manifest,
            split=args.split,
        )
        print(json.dumps(
            aggregate_by_method(
                rows,
                include_unavailable=args.include_unavailable,
            ),
            indent=2,
        ))
    else:
        print(format_report(evaluate_run(args.run_dir)))


if __name__ == "__main__":
    main()
