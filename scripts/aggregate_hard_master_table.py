#!/usr/bin/env python3
"""Aggregate MultiAgentBench Hard benchmark runs into Master Table 1.

Computes all 11 diagnostic metrics:
1. Task Score (mean)
2. Task Success Rate (%)
3. Total Tokens (k)
4. Episode Latency (s)
5. Active Memory Count
6. Memory Read Count
7. Private Read Count
8. Cross-Agent Reads
9. Memory Reuse Rate (%)
10. Negative Transfer (cross-agent reads in failed tasks)
11. Net Episode Reward
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_manifest(manifest_path: str) -> Dict[str, Any]:
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def get_target_tasks(manifest: Dict[str, Any], split: str = "test_hard") -> set[tuple[str, int]]:
    tasks = manifest.get("splits", {}).get(split, [])
    return {(str(t["benchmark"]), int(t["task_id"])) for t in tasks}


CANONICAL_ORDER = [
    "ours_rl",
    "ours_sft",
    "ours_base",
    "ours_private_to_global",
    "global_add_all",
    "no_memory",
    "single_agent",
    "mem0_style",
    "memory_r1_style",
    "g_memory_style",
    "collabmem_style",
    "copper_style",
    "lts_style",
]

NAME_DISPLAY = {
    "ours_rl": "Ours-RL (Governed)",
    "ours_sft": "Ours-SFT",
    "ours_base": "Ours-Base (Zero-Shot)",
    "ours_private_to_global": "Ours (Private->Global)",
    "global_add_all": "Global-Always",
    "no_memory": "No-Memory",
    "single_agent": "Single-Agent",
    "mem0_style": "Mem0 (Canonical)",
    "memory_r1_style": "Memory-R1",
    "g_memory_style": "G-Memory",
    "collabmem_style": "CollabMem",
    "copper_style": "CoPPER",
    "lts_style": "LTS",
}


def aggregate_runs(
    run_dirs: List[str],
    manifest_path: str,
    split: str = "test_hard",
) -> Dict[str, Any]:
    manifest = load_manifest(manifest_path)
    target_tasks = get_target_tasks(manifest, split)
    total_target_count = len(target_tasks)

    import os

    summaries = []
    for rd in run_dirs:
        summaries.extend(glob.glob(f"{rd}/**/summary.json", recursive=True))
    summaries.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)

    records_by_method = defaultdict(dict)  # method -> (bn, task_id) -> record

    for s_path in summaries:
        try:
            with open(s_path, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue

        method = d.get("method")
        ablation = d.get("ablation")
        if method == "ours_rl" and ablation and "private_to_global" in ablation:
            method = "ours_private_to_global"

        bn = str(d.get("benchmark", ""))
        tid = d.get("task_id")
        if tid is None or (bn, int(tid)) not in target_tasks:
            continue

        key = (bn, int(tid))
        d["_source_path"] = s_path
        existing = records_by_method[method].get(key)
        if not existing:
            records_by_method[method][key] = d
        elif d.get("status") == "ok":
            # Prefer hard_* runs or newer runs
            existing_is_hard = "hard_" in existing.get("_source_path", "")
            current_is_hard = "hard_" in s_path
            if current_is_hard or not existing_is_hard:
                records_by_method[method][key] = d

    all_methods = [m for m in CANONICAL_ORDER if m in records_by_method]
    for m in records_by_method:
        if m not in all_methods:
            all_methods.append(m)

    table_rows = []
    for m in all_methods:
        task_map = records_by_method[m]
        recs = list(task_map.values())
        n = len(recs)
        if n == 0:
            continue

        scores = [float(r.get("task_score", 0.0) or 0.0) for r in recs]
        successes = [1 if r.get("task_success") else 0 for r in recs]
        tokens = [int(r.get("total_tokens", 0) or 0) for r in recs]
        latencies = [float(r.get("episode_latency_s", 0.0) or 0.0) for r in recs]
        rewards = [float(r.get("episode_reward", 0.0) or 0.0) for r in recs]

        mem_counts = []
        priv_reads = []
        cross_reads = []
        total_reads = []
        reuse_rates = []
        neg_transfers = []

        ctrl_models = set()

        for r in recs:
            mm = r.get("memory_metrics") or {}
            is_success = bool(r.get("task_success"))
            active_cnt = mm.get("active_memory_count", 0)
            mem_counts.append(active_cnt)

            cr = mm.get("cross_agent_reads", 0)
            cross_reads.append(cr)

            reads = mm.get("reads", 0)
            total_reads.append(reads)

            pr = mm.get("targeted_read", mm.get("private_read", max(reads - cr, 0)))
            priv_reads.append(pr)

            rr = mm.get("reuse_rate", 0.0)
            reuse_rates.append(rr)

            neg_transfers.append(cr if not is_success else 0)

            cm = r.get("controller_model") or r.get("controller") or ""
            if cm:
                ctrl_models.add(cm)

        row = {
            "method": m,
            "display_name": NAME_DISPLAY.get(m, m),
            "completed": f"{n}/{total_target_count}",
            "task_score": sum(scores) / n,
            "task_success_rate": (sum(successes) / n) * 100.0,
            "tokens_k": (sum(tokens) / n) / 1000.0,
            "latency_s": sum(latencies) / n,
            "active_memories": sum(mem_counts) / n,
            "private_reads": sum(priv_reads) / n,
            "cross_reads": sum(cross_reads) / n,
            "reuse_rate_pct": (sum(reuse_rates) / n) * 100.0 if any(reuse_rates) else 0.0,
            "negative_transfer": sum(neg_transfers) / n,
            "net_reward": sum(rewards) / n,
            "controller_info": "; ".join(sorted(ctrl_models)) or "standard",
        }
        table_rows.append(row)

    return {
        "split": split,
        "manifest": manifest_path,
        "target_tasks": total_target_count,
        "rows": table_rows,
    }


def print_markdown_table(result: Dict[str, Any]) -> str:
    rows = result["rows"]
    if not rows:
        return "No completed episode records found."

    lines = []
    lines.append(f"### MARBLE Master Table 1: Benchmark Results ({result['split']}, N={result['target_tasks']})")
    lines.append("")
    lines.append(
        "| Method | Tasks | Task Score | Success (%) | Tokens (k) | Latency (s) | Active Mem | Priv Reads | Cross Reads | Reuse (%) | Neg Transfer | Net Reward | Controller Model |"
    )
    lines.append(
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|"
    )

    for r in rows:
        lines.append(
            f"| **{r['display_name']}** | {r['completed']} | {r['task_score']:.3f} | {r['task_success_rate']:.1f}% | {r['tokens_k']:.1f}k | {r['latency_s']:.1f}s | {r['active_memories']:.1f} | {r['private_reads']:.2f} | {r['cross_reads']:.2f} | {r['reuse_rate_pct']:.1f}% | {r['negative_transfer']:.2f} | {r['net_reward']:.3f} | `{r['controller_info']}` |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate MARBLE Hard benchmark results.")
    parser.add_argument("--run-dir", nargs="+", required=True, help="Directory containing run outputs")
    parser.add_argument(
        "--manifest",
        default="configs/experiments/multiagentbench_hard_frozen.json",
        help="Path to experiment manifest",
    )
    parser.add_argument("--split", default="test_hard", help="Split name to evaluate")
    parser.add_argument("--out-json", default=None, help="Save aggregated JSON table")
    args = parser.parse_args()

    res = aggregate_runs(args.run_dir, args.manifest, split=args.split)
    md = print_markdown_table(res)
    print(md)

    if args.out_json:
        out_p = Path(args.out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"\nSaved JSON table to: {args.out_json}")


if __name__ == "__main__":
    main()
