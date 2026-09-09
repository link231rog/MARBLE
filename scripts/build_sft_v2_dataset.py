#!/usr/bin/env python3
"""Build and audit rebuilt SFT v2 dataset supporting absent, global, and targeted.

Adheres strictly to Chapter 7 Action Guide:
- Covers full frozen action space: absent, global, targeted (with explicit recipient IDs).
- Only uses train_hard tasks (0 overlap with test_hard).
- Validates that invalid recipient count == 0, parse error count == 0.
- Exports training pairs compatible with export_sft_pairs and train_qwen_sft.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from marble.controllers.qwen_lora import export_sft_pairs, audit_sft_distribution
from marble.controllers.json_controller import TOPIC_TAXONOMY


def upgrade_prompt_to_v2(prompt: str, arm: Dict[str, str]) -> str:
    """Rewrite legacy prompt to match frozen first-class action schema."""
    agent_ids = sorted(arm.keys()) if arm else ["agent1", "agent2"]
    example_agent_1 = f'["{agent_ids[0]}"]' if agent_ids else '["agent_1"]'
    example_agent_2 = (
        f'["{agent_ids[0]}", "{agent_ids[1]}"]'
        if len(agent_ids) >= 2
        else '["agent_1", "agent_2"]'
    )
    system_block_replacement = (
        f"[SYSTEM]\n"
        f"agent_role_map: {json.dumps(arm, ensure_ascii=False, sort_keys=True)}\n"
        f"topic_taxonomy: {list(TOPIC_TAXONOMY)}\n"
        f"visibility: absent removes memory; global exposes to all agents; a JSON array of agent IDs (e.g. "
        f"{example_agent_1} or {example_agent_2}) from agent_role_map routes strictly to those agents.\n"
        f"update: use an exact active memory_id only when updating that memory.\n"
        f'output: ONLY {{"visibility": "absent"|"global"|["<agent_id>", ...], "update": null}}.'
    )
    task_idx = prompt.find("[TASK]")
    if task_idx != -1:
        rest = prompt[task_idx:]
    else:
        rest = prompt
    return f"{system_block_replacement}\n{rest}"


def classify_proposal(
    benchmark: str,
    task_id: str,
    proposal: Dict[str, Any],
    arm: Dict[str, str],
) -> Dict[str, Any]:
    """Expert routing classification for a proposal in a train_hard task."""
    agent_id = proposal.get("agent_id", "")
    title = (proposal.get("title") or "").lower()
    raw = (proposal.get("raw_value") or "").lower()
    known_agents = sorted(arm.keys()) if arm else []

    # 1. Obvious ABSENT noise: syntax errors, connection failures, empty query results
    if (
        not title
        or len(raw) < 25
        or "syntax error" in raw
        or "could not connect" in raw
        or "no rows" in raw
        or "internal server error" in raw
    ):
        return {"visibility": "absent", "supersedes": None, "target_recipients": []}

    # 2. Domain: Coding (Tasks 47, 51, 52, 53)
    if benchmark == "coding":
        # agent1: developer, agent2: reviewer/optimizer, agent3: tester
        if agent_id == "agent2":
            # Code review / bug advice
            if "test" in title or "test" in raw or "edge case" in raw:
                recipients = [a for a in ["agent1", "agent3"] if a in arm]
            else:
                recipients = ["agent1"] if "agent1" in arm else known_agents[:1]
            return {"visibility": "targeted", "supersedes": None, "target_recipients": recipients}
        elif agent_id == "agent3":
            # Tester feedback / bug findings
            if "unit test" in title or "pytest" in raw or "failure" in raw or "assertion" in raw:
                recipients = ["agent1"] if "agent1" in arm else known_agents[:1]
                return {"visibility": "targeted", "supersedes": None, "target_recipients": recipients}
            elif "benchmark" in title or "performance" in title or "optimiz" in title:
                recipients = [a for a in ["agent1", "agent2"] if a in arm]
                return {"visibility": "targeted", "supersedes": None, "target_recipients": recipients}
        elif agent_id == "agent1":
            # Developer implementation
            if "solution file" in title or "complete" in title or "architecture" in title or "cli" in title:
                return {"visibility": "global", "supersedes": None, "target_recipients": []}
            elif "module" in title or "helper" in title or "class" in title or "function" in title:
                recipients = [a for a in ["agent2", "agent3"] if a in arm]
                return {"visibility": "targeted", "supersedes": None, "target_recipients": recipients}
        return {"visibility": "global", "supersedes": None, "target_recipients": []}

    # 3. Domain: Database (Tasks 51, 52, 53, 54)
    elif benchmark == "database":
        is_lock = "lock" in title or "lock" in raw or "blocked" in raw or "deadlock" in raw
        is_vacuum = "vacuum" in title or "vacuum" in raw or "dead tuple" in raw or "bloat" in raw
        is_index = "index" in title or "index" in raw or "seq scan" in raw or "scan" in title
        is_fetch = "fetch" in title or "select" in title or "slow query" in raw or "large data" in raw
        is_insert = "insert" in title or "copy" in raw or "bulk" in raw

        # Multi-recipient targeted combinations
        if is_lock and is_insert:
            recipients = [a for a in ["agent1", "agent2"] if a in arm]
            return {"visibility": "targeted", "supersedes": None, "target_recipients": recipients}
        if is_index and is_fetch:
            recipients = [a for a in ["agent4", "agent5"] if a in arm]
            return {"visibility": "targeted", "supersedes": None, "target_recipients": recipients}

        # Single-recipient targeted
        if is_lock and "agent2" in arm and agent_id != "agent2":
            return {"visibility": "targeted", "supersedes": None, "target_recipients": ["agent2"]}
        if is_vacuum and "agent3" in arm and agent_id != "agent3":
            return {"visibility": "targeted", "supersedes": None, "target_recipients": ["agent3"]}
        if is_index and "agent4" in arm and agent_id != "agent4":
            return {"visibility": "targeted", "supersedes": None, "target_recipients": ["agent4"]}
        if is_fetch and "agent5" in arm and agent_id != "agent5":
            return {"visibility": "targeted", "supersedes": None, "target_recipients": ["agent5"]}
        if is_insert and "agent1" in arm and agent_id != "agent1":
            return {"visibility": "targeted", "supersedes": None, "target_recipients": ["agent1"]}

        # Global consensus / overview
        if "summary" in title or "overview" in title or "root cause" in title or "tps" in raw or "pg_stat" in raw:
            return {"visibility": "global", "supersedes": None, "target_recipients": []}
        return {"visibility": "absent", "supersedes": None, "target_recipients": []}

    # 4. Domain: Research (Tasks 10, 11, 15, 29)
    elif benchmark == "research":
        # Global synthesized proposals
        if "5q" in title or "5-question" in title or "proposal" in title or "research plan" in title or "hypothesis" in title:
            return {"visibility": "global", "supersedes": None, "target_recipients": []}

        # Sub-topic targeting
        if "privacy" in title or "speech" in title or "audio" in title or "emotion" in title or "ser" in raw:
            recipients = [a for a in ["agent2", "agent4"] if a in arm]
            if recipients:
                return {"visibility": "targeted", "supersedes": None, "target_recipients": recipients}
        if "vision" in title or "video" in title or "image" in title or "clip" in raw:
            recipients = [a for a in ["agent3", "agent4", "agent8"] if a in arm]
            if recipients:
                return {"visibility": "targeted", "supersedes": None, "target_recipients": recipients}
        if "graph" in title or "gnn" in title or "math" in title or "relation" in raw or "algebra" in raw:
            recipients = [a for a in ["agent1", "agent2"] if a in arm]
            if recipients:
                return {"visibility": "targeted", "supersedes": None, "target_recipients": recipients}
        if "pruning" in title or "compression" in title or "efficient" in title or "quantization" in raw:
            recipients = [a for a in ["agent5", "agent6"] if a in arm]
            if recipients:
                return {"visibility": "targeted", "supersedes": None, "target_recipients": recipients}
        if "paper" in title or "literature" in title:
            other_agents = [a for a in known_agents if a != agent_id]
            if other_agents:
                return {"visibility": "targeted", "supersedes": None, "target_recipients": [other_agents[0]]}
        return {"visibility": "absent", "supersedes": None, "target_recipients": []}

    return {"visibility": "absent", "supersedes": None, "target_recipients": []}


def build_sft_v2_dataset(
    manifest_path: str = "configs/experiments/multiagentbench_hard_frozen.json",
    train_traces_dir: str = "runs/train_hard_traces",
    out_dir: str = "runs/sft_rebuild_smoke_20260909",
) -> Dict[str, Any]:
    """Rebuild SFT v2 dataset adhering strictly to Chapter 7 action guide."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen manifest and verify train/test split keys
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    train_hard_keys = set((item["benchmark"], int(item["task_id"])) for item in manifest["splits"]["train_hard"])
    test_hard_keys = set((item["benchmark"], int(item["task_id"])) for item in manifest["splits"]["test_hard"])

    # Hard check: zero overlap
    overlap = train_hard_keys.intersection(test_hard_keys)
    if overlap:
        raise ValueError(f"FATAL: train_hard and test_hard have overlapping tasks: {overlap}")

    # 2. Find all train_hard traces
    all_traces = sorted(glob.glob(f"{train_traces_dir}/**/memory_trace.jsonl", recursive=True))
    if not all_traces:
        raise FileNotFoundError(f"No memory_trace.jsonl files found in {train_traces_dir}")

    rebuilt_events: List[Dict[str, Any]] = []
    task_keys_seen: Set[Tuple[str, int]] = set()

    total_samples = 0
    visibility_counts = {"absent": 0, "global": 0, "targeted": 0}
    recipient_set_counts: Dict[str, int] = {}
    invalid_recipient_count = 0
    parse_schema_error_count = 0

    for trace_file in all_traces:
        with open(trace_file, encoding="utf-8") as tf:
            # First pass: find agent_role_map
            arm: Dict[str, str] = {}
            for line in tf:
                if not line.strip():
                    continue
                d = json.loads(line)
                prompt = d.get("controller_prompt", "")
                if "agent_role_map:" in prompt:
                    for pline in prompt.split("\n"):
                        if pline.startswith("agent_role_map:"):
                            arm = json.loads(pline[len("agent_role_map:") :].strip())
                            break
                    if arm:
                        break

            # Second pass: process memory_decision events
            tf.seek(0)
            parts = Path(trace_file).parts
            # Determine benchmark from directory structure
            bench = ""
            for candidate in ("database", "research", "coding"):
                if candidate in parts:
                    bench = candidate
                    break

            for line in tf:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("event") != "memory_decision":
                    continue

                prop = d.get("proposal", {})
                task_id_str = str(prop.get("task_id", ""))
                try:
                    task_id = int(task_id_str)
                except ValueError:
                    continue

                task_key = (bench, task_id)
                # Verify strictly belongs to train_hard
                if task_key not in train_hard_keys:
                    continue
                task_keys_seen.add(task_key)

                # Classify proposal
                cls = classify_proposal(bench, task_id_str, prop, arm)
                vis = cls["visibility"]
                recipients = sorted(cls.get("target_recipients") or [])

                # Validation checks
                if vis == "targeted":
                    if not recipients:
                        invalid_recipient_count += 1
                    for r in recipients:
                        if arm and r not in arm:
                            invalid_recipient_count += 1
                    rec_key = str(tuple(recipients))
                    recipient_set_counts[rec_key] = recipient_set_counts.get(rec_key, 0) + 1

                visibility_counts[vis] += 1
                total_samples += 1

                # Upgrade controller_prompt to ensure 100% alignment with current action schema
                prompt_raw = d.get("controller_prompt")
                prompt_aligned = upgrade_prompt_to_v2(prompt_raw, arm) if prompt_raw else ""

                # Construct updated event
                updated_ev = {
                    "event": "memory_decision",
                    "memory_id": d.get("memory_id", f"mem_{prop.get('proposal_id', total_samples)}"),
                    "proposal": prop,
                    "target": {
                        "visibility": "targeted" if vis == "targeted" else vis,
                        "target_recipients": recipients if vis == "targeted" else [],
                        "supersedes": None,
                    },
                    "controller_prompt": prompt_aligned,
                }
                rebuilt_events.append(updated_ev)

    # Hard assert on Chapter 7 requirements
    train_test_task_overlap_count = len(task_keys_seen.intersection(test_hard_keys))
    if train_test_task_overlap_count != 0:
        raise ValueError(f"FATAL: train/test task overlap detected: {train_test_task_overlap_count}")
    if invalid_recipient_count != 0:
        raise ValueError(f"FATAL: invalid recipient count must be 0, got {invalid_recipient_count}")

    # Audit prompt schema alignment across all events
    prompt_schema_mismatch_count = 0
    for ev in rebuilt_events:
        p = ev.get("controller_prompt", "")
        if "private exposes only to owner" in p or 'ONLY {"visibility": "absent"|"private"|"global"' in p:
            prompt_schema_mismatch_count += 1
        if 'output: ONLY {"visibility": "absent"|"global"|["<agent_id>", ...]' not in p:
            prompt_schema_mismatch_count += 1

    if prompt_schema_mismatch_count != 0:
        raise ValueError(f"FATAL: prompt schema mismatch count must be 0, got {prompt_schema_mismatch_count}")

    # Write rebuilt trace file
    trace_out_path = out_path / "sft_v2_trace.jsonl"
    with open(trace_out_path, "w", encoding="utf-8") as out_fh:
        for ev in rebuilt_events:
            out_fh.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # Export SFT pairs via official marble export_sft_pairs
    sft_pairs = export_sft_pairs([str(trace_out_path)])
    if len(sft_pairs) != total_samples:
        raise RuntimeError(f"Exported pairs count ({len(sft_pairs)}) != total samples ({total_samples})")

    # Validate all exported completions
    for _, completion in sft_pairs:
        try:
            parsed = json.loads(completion)
            if "visibility" not in parsed or ("update" not in parsed and "supersedes" not in parsed):
                parse_schema_error_count += 1
        except Exception:
            parse_schema_error_count += 1

    if parse_schema_error_count != 0:
        raise ValueError(f"FATAL: parse/schema error count must be 0, got {parse_schema_error_count}")

    # Distribution audit via marble audit_sft_distribution
    audit = audit_sft_distribution(sft_pairs)

    stats = {
        "total_samples": total_samples,
        "visibility_counts": visibility_counts,
        "target_recipient_set_counts": recipient_set_counts,
        "invalid_recipient_count": invalid_recipient_count,
        "parse_schema_error_count": parse_schema_error_count,
        "train_test_task_overlap_count": train_test_task_overlap_count,
        "prompt_schema_mismatch_count": prompt_schema_mismatch_count,
        "train_hard_tasks_covered": sorted([f"{b}_{tid}" for b, tid in task_keys_seen]),
        "audit_distribution": audit,
    }

    # Save stats JSON
    stats_out_path = out_path / "dataset_stats.json"
    with open(stats_out_path, "w", encoding="utf-8") as sfh:
        json.dump(stats, sfh, indent=2, ensure_ascii=False)

    # Save human-readable audit report
    audit_report_path = out_path / "audit_report.txt"
    report_lines = [
        "======================================================================",
        "MARBLE Chapter 7: Rebuilt SFT v2 Dataset Audit Report",
        "======================================================================",
        f"Trace output file:              {trace_out_path}",
        f"Total training samples:         {total_samples}",
        "",
        "Visibility Distribution:",
        f"  ABSENT:                       {visibility_counts['absent']} ({visibility_counts['absent']/total_samples*100:.1f}%)",
        f"  GLOBAL:                       {visibility_counts['global']} ({visibility_counts['global']/total_samples*100:.1f}%)",
        f"  TARGETED:                     {visibility_counts['targeted']} ({visibility_counts['targeted']/total_samples*100:.1f}%)",
        "",
        f"Targeted Recipient Sets ({len(recipient_set_counts)} distinct combinations):",
    ]
    for rset, count in sorted(recipient_set_counts.items(), key=lambda x: -x[1]):
        report_lines.append(f"  {rset:30s}: {count:4d} samples")

    report_lines.extend([
        "",
        "Chapter 7 Section 4 Hard Checks:",
        f"  Invalid Recipient Count:      {invalid_recipient_count} (Must be 0) -> {'PASSED ✅' if invalid_recipient_count == 0 else 'FAILED ❌'}",
        f"  Parse / Schema Error Count:   {parse_schema_error_count} (Must be 0) -> {'PASSED ✅' if parse_schema_error_count == 0 else 'FAILED ❌'}",
        f"  Train / Test Task Overlap:    {train_test_task_overlap_count} (Must be 0) -> {'PASSED ✅' if train_test_task_overlap_count == 0 else 'FAILED ❌'}",
        f"  Prompt Schema Mismatch Count: {prompt_schema_mismatch_count} (Must be 0) -> {'PASSED ✅' if prompt_schema_mismatch_count == 0 else 'FAILED ❌'}",
        f"  ABSENT > 0:                   {'PASSED ✅' if visibility_counts['absent'] > 0 else 'FAILED ❌'}",
        f"  GLOBAL > 0:                   {'PASSED ✅' if visibility_counts['global'] > 0 else 'FAILED ❌'}",
        f"  TARGETED > 0:                 {'PASSED ✅' if visibility_counts['targeted'] > 0 else 'FAILED ❌'}",
        "======================================================================",
    ])

    with open(audit_report_path, "w", encoding="utf-8") as arf:
        arf.write("\n".join(report_lines) + "\n")

    print("\n".join(report_lines))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SFT v2 dataset.")
    parser.add_argument("--manifest", default="configs/experiments/multiagentbench_hard_frozen.json")
    parser.add_argument("--train-traces", default="runs/train_hard_traces")
    parser.add_argument("--out", default="runs/sft_rebuild_smoke_20260909")
    args = parser.parse_args()

    build_sft_v2_dataset(args.manifest, args.train_traces, args.out)
