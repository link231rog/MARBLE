#!/usr/bin/env python3
"""Aggregate controller parameter scaling analysis results (0.8B, 2B, 4B, 9B) across Base, SFT, GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _extract_controller_cost(trace_path: Path) -> Dict[str, int]:
    if not trace_path.is_file():
        return {"calls": 0, "tokens": 0}
    calls = 0
    tokens = 0
    try:
        with trace_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                ev = json.loads(line)
                if ev.get("event") == "memory_decision":
                    calls += 1
                    p = ev.get("controller_prompt", "")
                    o = ev.get("controller_output", "")
                    tokens += max(1, len(p) // 3) + max(1, len(o) // 3)
    except Exception:
        pass
    return {"calls": calls, "tokens": tokens}


def load_run_summaries(run_dir: Path, target_seeds: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    summaries = []
    if not run_dir.is_dir():
        return summaries
    for p in sorted(run_dir.rglob("summary.json")):
        try:
            with p.open(encoding="utf-8") as fh:
                d = json.load(fh)
                if d.get("status") == "ok":
                    if target_seeds is not None:
                        s = d.get("seed")
                        if s is not None and s not in target_seeds:
                            continue
                    trace_path = p.parent / "memory_trace.jsonl"
                    cost = _extract_controller_cost(trace_path)
                    d["ctrl_calls"] = cost["calls"]
                    d["ctrl_tokens"] = cost["tokens"]
                    summaries.append(d)
        except Exception:
            pass
    return summaries


def compute_metrics(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    detected_seeds = sorted(list(set(d["seed"] for d in summaries if "seed" in d)))
    if not summaries:
        return {
            "task_count": 0,
            "detected_seeds": detected_seeds,
            "success_rate": 0.0,
            "mean_score": 0.0,
            "active_memories": 0.0,
            "cross_agent_reads": 0.0,
            "productive_cross_reads": 0.0,
            "productive_cross_read_rate": 0.0,
            "negative_transfer": 0.0,
            "ctrl_calls": 0.0,
            "ctrl_tokens_k": 0.0,
        }

    n = len(summaries)
    successes = sum(1 for d in summaries if bool(d.get("task_success", False)))
    scores = sum(float(d.get("task_score", 0.0)) for d in summaries)

    act_mems = 0.0
    cross_reads = 0.0
    prod_reads = 0.0
    prod_rates = 0.0
    neg_trans = 0.0
    ctrl_calls = 0.0
    ctrl_tokens = 0.0

    for d in summaries:
        mm = d.get("memory_metrics") or {}
        act_mems += float(mm.get("active_memory_count", d.get("active_memory_count", 0)))
        cr = float(mm.get("cross_agent_reads", d.get("cross_agent_reads", 0)))
        pr = float(mm.get("productive_cross_reads", d.get("productive_cross_reads", 0)))
        pr_rate = float(mm.get("productive_cross_read_rate", d.get("productive_cross_read_rate", 0.0)))
        nt = float(mm.get("negative_transfer", d.get("negative_transfer", 0.0)))
        ctrl_calls += float(d.get("ctrl_calls", 0))
        ctrl_tokens += float(d.get("ctrl_tokens", 0))

        cross_reads += cr
        prod_reads += pr
        prod_rates += pr_rate
        neg_trans += nt

    return {
        "task_count": n,
        "detected_seeds": detected_seeds,
        "success_rate": successes / n,
        "mean_score": scores / n,
        "active_memories": act_mems / n,
        "cross_agent_reads": cross_reads / n,
        "productive_cross_reads": prod_reads / n,
        "productive_cross_read_rate": prod_rates / n,
        "negative_transfer": neg_trans / n,
        "ctrl_calls": ctrl_calls / n,
        "ctrl_tokens_k": (ctrl_tokens / n) / 1000.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate controller parameter scaling results.")
    parser.add_argument("--base-dir", default="runs/controller_scaling_analysis",
                        help="Base directory containing scale subdirectories (0.8b, 2b, 4b, 9b)")
    parser.add_argument("--scales", nargs="+", default=["0.8b", "2b", "4b", "9b"],
                        help="Model scales to aggregate")
    parser.add_argument("--methods", nargs="+", default=["ours_base", "ours_sft", "ours_rl"],
                        help="Methods to aggregate per scale")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Optional specific seed filter (e.g. --seeds 42 43)")
    parser.add_argument("--json-out", default=None, help="Optional output path for JSON report")
    args = parser.parse_args()

    base_path = Path(args.base_dir)
    results = {}

    print("\n" + "=" * 125)
    print("📈 MARBLE Controller Parameter Scaling Analysis (Capacity & Cost Sensitivity)")
    print("=" * 125)
    print(f"{'Scale':<8} {'Method':<10} {'Seeds':<8} {'Tasks':<6} {'Success':<9} {'Score':<7} {'CtrlCalls':<11} {'CtrlTokens':<12} {'Useful%':<9} {'NegTrans%':<10}")
    print("-" * 125)

    for scale in args.scales:
        scale_key = scale.lower()
        results[scale_key] = {}
        for method in args.methods:
            run_dir = base_path / scale_key / method
            # Fallback to existing 4b runs if not populated in scaling dir yet
            if not run_dir.exists() and scale_key == "4b":
                if method == "ours_sft" and Path("runs/rl_smoke_check/seed_42/ours_sft").exists():
                    run_dir = Path("runs/rl_smoke_check/seed_42/ours_sft")

            summaries = load_run_summaries(run_dir, target_seeds=args.seeds)
            metrics = compute_metrics(summaries)
            results[scale_key][method] = metrics

            disp_method = {"ours_base": "Base", "ours_sft": "SFT", "ours_rl": "GRPO"}.get(method, method)
            detected = metrics.get("detected_seeds", [])
            seeds_str = ",".join(str(s) for s in detected) if detected else "-"
            tasks_str = str(metrics["task_count"])
            succ_str = f"{metrics['success_rate'] * 100:.1f}%"
            score_str = f"{metrics['mean_score']:.3f}"
            calls_str = f"{metrics['ctrl_calls']:.1f}"
            tokens_str = f"{metrics['ctrl_tokens_k']:.1f}k"
            useful_str = f"{metrics['productive_cross_read_rate'] * 100:.1f}%"
            neg_str = f"{metrics['negative_transfer'] * 100:.1f}%"

            print(f"{scale.upper():<8} {disp_method:<10} {seeds_str:<8} {tasks_str:<6} {succ_str:<9} {score_str:<7} {calls_str:<11} {tokens_str:<12} {useful_str:<9} {neg_str:<10}")

    print("=" * 125 + "\n")

    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"✅ JSON summary saved to {out_p}")


if __name__ == "__main__":
    main()
