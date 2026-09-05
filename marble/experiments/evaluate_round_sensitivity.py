"""Aggregates round/iteration sensitivity sweep results across horizons T."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

from marble.experiments.evaluate import aggregate_by_method, evaluate_run_root


def parse_round_from_dir(path: str) -> tuple[int, str]:
    """Extracts (T, baseline) from e.g. runs/sensitivity_rounds_T4_ours_rl."""
    match = re.search(r"_T(\d+)_(.+)$", os.path.basename(path))
    if match:
        return int(match.group(1)), match.group(2)
    return 0, os.path.basename(path)


def main():
    parser = argparse.ArgumentParser(description="Aggregate iteration sensitivity sweep.")
    parser.add_argument("--pattern", default="runs/sensitivity_rounds_T*", help="Glob pattern for sweep dirs")
    parser.add_argument("--manifest", default="configs/experiments/multiagentbench_hard_frozen.json")
    parser.add_argument("--split", default="test_hard")
    parser.add_argument("--out", default="runs/round_sensitivity_analysis_table.json")
    args = parser.parse_args()

    dirs = sorted(glob.glob(args.pattern), key=lambda d: parse_round_from_dir(d))
    results: Dict[str, Dict[str, Any]] = {}

    for d in dirs:
        if not os.path.isdir(d):
            continue
        T, baseline = parse_round_from_dir(d)
        t_key = f"T={T}"
        results.setdefault(t_key, {})
        rows = evaluate_run_root(d, manifest=args.manifest, split=args.split)
        if rows:
            agg = aggregate_by_method(rows)
            # Find the main method metrics
            method_key = next(iter(agg.keys()), baseline)
            results[t_key][baseline] = {
                "setting": method_key,
                "task_score": agg[method_key].get("task_score", 0.0),
                "task_score_se": agg[method_key].get("task_score_se", 0.0),
                "total_tokens": agg[method_key].get("total_tokens", 0.0),
                "total_tokens_se": agg[method_key].get("total_tokens_se", 0.0),
                "active_memory_count": agg[method_key].get("memory.active_memory_count", 0.0),
                "active_private_count": agg[method_key].get("memory.active_private_count", 0.0),
                "active_global_count": agg[method_key].get("memory.active_global_count", 0.0),
                "private_decisions": agg[method_key].get("memory.private", 0.0),
                "global_decisions": agg[method_key].get("memory.global", 0.0),
                "supersessions": agg[method_key].get("memory.supersessions", 0.0),
                "cross_agent_reads": agg[method_key].get("memory.cross_agent_reads", 0.0),
                "reuse_rate": agg[method_key].get("memory.reuse_rate", 0.0),
            }
            # Calculate private ratio if any decisions
            priv = results[t_key][baseline]["private_decisions"]
            glob_cnt = results[t_key][baseline]["global_decisions"]
            denom = priv + glob_cnt
            results[t_key][baseline]["private_ratio"] = priv / denom if denom > 0 else 0.0

    print(json.dumps(results, indent=2))
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n✅ Sensitivity analysis saved to {args.out}")


if __name__ == "__main__":
    main()
