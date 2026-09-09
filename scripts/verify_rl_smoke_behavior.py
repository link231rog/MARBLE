import json
import os
import sys
from pathlib import Path
from marble.experiments.evaluate import evaluate_memory_trace
from marble.controllers.qwen_lora import load_qwen_rl_samples

def main():
    rollout_dir = Path("runs/rl_smoke_check")
    traces = sorted(rollout_dir.glob("**/memory_trace.jsonl"))
    print(f"Found {len(traces)} traces in {rollout_dir}")
    if not traces:
        print(f"Error: no traces found in {rollout_dir}")
        sys.exit(1)

    all_metrics = []
    total_decisions = 0
    vis_counts = {"absent": 0, "private": 0, "targeted": 0, "global": 0}
    total_cross_reads = 0
    total_prod_cross_reads = 0
    scores = []
    valid_traces = []

    for t in traces:
        s_path = t.with_name("summary.json")
        if not s_path.is_file():
            continue
        with s_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        score = float(summary.get("task_score", 0.0) or 0.0)
        success = summary.get("task_success")

        with t.open(encoding="utf-8") as f:
            events = [json.loads(line) for line in f if line.strip()]

        m = evaluate_memory_trace(events, task_success=success, task_score=score)
        all_metrics.append(m)
        scores.append(score)
        valid_traces.append(str(t.resolve()))

        total_decisions += m.get("proposals_total", 0)
        total_cross_reads += m.get("cross_agent_reads", 0)
        total_prod_cross_reads += m.get("productive_cross_reads", 0)

        for ev in events:
            if ev.get("event") == "memory_decision":
                vis = ev.get("target", {}).get("visibility", "unknown")
                vis_counts[vis] = vis_counts.get(vis, 0) + 1

    print("\n========================================================")
    print("📊 SMOKE CHECK 1: Visibility Output Distribution")
    print("========================================================")
    print(f"Total proposals/decisions evaluated: {total_decisions}")
    for vis, count in sorted(vis_counts.items()):
        pct = (count / total_decisions * 100) if total_decisions > 0 else 0
        print(f"  {vis.upper():<10}: {count:>4} ({pct:.1f}%)")

    has_absent = vis_counts.get("absent", 0) > 0
    has_private = (vis_counts.get("private", 0) + vis_counts.get("targeted", 0)) > 0
    has_global = vis_counts.get("global", 0) > 0
    # In collaborative tasks (DB, Coding, Research), ABSENT (filter) and GLOBAL (share) are the essential modalities.
    criterion_1 = has_absent and has_global
    print(f"\n  -> Criterion 1 (Active filtering & sharing: ABSENT={has_absent}, GLOBAL={has_global}): {'✅ PASSED' if criterion_1 else '❌ FAILED'}")

    print("\n========================================================")
    print("📊 SMOKE CHECK 2: Collaboration & Productive Reads")
    print("========================================================")
    n_episodes = max(len(valid_traces), 1)
    mean_cr = total_cross_reads / n_episodes
    mean_pcr = total_prod_cross_reads / n_episodes
    pcr_rate = (total_prod_cross_reads / total_cross_reads * 100) if total_cross_reads > 0 else 0
    active_mems = sum(m.get("active_memory_count", 0) for m in all_metrics) / n_episodes

    print(f"  Total Raw Cross-Agent Reads:    {total_cross_reads} (mean: {mean_cr:.2f})")
    print(f"  Total Productive Cross Reads:   {total_prod_cross_reads} (mean: {mean_pcr:.2f})")
    print(f"  Productive Cross Read Rate:     {pcr_rate:.1f}%")
    print(f"  Mean Active Memory Count:       {active_mems:.2f}")

    criterion_2 = total_cross_reads > 0
    criterion_3 = total_prod_cross_reads > 0
    criterion_4 = active_mems >= 0.5

    print(f"\n  -> Criterion 2 (Cross-agent reads > 0): {'✅ PASSED' if criterion_2 else '❌ FAILED'}")
    print(f"  -> Criterion 3 (Productive cross reads > 0): {'✅ PASSED' if criterion_3 else '❌ FAILED'}")
    print(f"  -> Criterion 4 (Active memory not collapsed, mean={active_mems:.2f}): {'✅ PASSED' if criterion_4 else '❌ FAILED'}")

    print("\n========================================================")
    print("📊 SMOKE CHECK 3: Credit Assignment Sample Validation")
    print("========================================================")
    samples, stats = load_qwen_rl_samples(valid_traces, scores)
    print(f"  Traces loaded:            {stats['traces']}")
    print(f"  Decisions processed:      {stats['decisions']}")
    print(f"  Valid RL samples:         {stats['samples']}")
    print(f"  Skipped format error:     {stats.get('skipped_format_error', 0)}")

    adv_by_vis = {}
    for s in samples:
        try:
            d = json.loads(s["completion"])
            v = str(d.get("visibility", "unknown"))
        except:
            v = "err"
        adv_by_vis.setdefault(v, []).append(s["advantage"])

    print("\n  Sample Advantage by Visibility:")
    for v, vals in sorted(adv_by_vis.items()):
        mean_v = sum(vals) / len(vals)
        print(f"    {v.upper():<10}: N={len(vals):>3}, mean_advantage={mean_v:>7.4f}, range=[{min(vals):.4f}, {max(vals):.4f}]")

    all_passed = criterion_1 and criterion_2 and criterion_3 and criterion_4
    print("\n========================================================")
    if all_passed:
        print("🎉 ALL 4 BEHAVIOR CRITERIA PASSED! Ready for 1 GRPO update.")
    else:
        print("⚠️ SOME CRITERIA FAILED. Please review diagnostic output.")
    print("========================================================")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
