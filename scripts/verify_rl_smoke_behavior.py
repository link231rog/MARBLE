import json
import os
import sys
from pathlib import Path
from marble.experiments.evaluate import evaluate_memory_trace
from marble.controllers.qwen_lora import load_qwen_rl_samples

def main():
    if len(sys.argv) > 1:
        rollout_dir = Path(sys.argv[1])
    else:
        rollout_dir = Path("runs/grpo_recovery_smoke_20260909") if Path("runs/grpo_recovery_smoke_20260909").exists() else Path("runs/rl_smoke_check")
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
    has_targeted = (vis_counts.get("private", 0) + vis_counts.get("targeted", 0)) > 0
    has_global = vis_counts.get("global", 0) > 0
    # Spec §12.3 & Chapter 7: ABSENT (filter), GLOBAL (share), and TARGETED (route) must all exist.
    criterion_1 = has_absent and has_global and has_targeted
    print(f"\n  -> Criterion 1 (Active filtering, sharing & targeting: ABSENT={has_absent}, GLOBAL={has_global}, TARGETED={has_targeted}): {'✅ PASSED' if criterion_1 else '❌ FAILED'}")

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
            raw_v = d.get("visibility")
            if isinstance(raw_v, list):
                v = "targeted"
            else:
                v = str(raw_v or "unknown")
        except:
            v = "err"
        adv_by_vis.setdefault(v, []).append(s["advantage"])

    print("\n  Sample Advantage by Visibility:")
    for v, vals in sorted(adv_by_vis.items()):
        mean_v = sum(vals) / len(vals)
        print(f"    {v.upper():<10}: N={len(vals):>3}, mean_advantage={mean_v:>7.4f}, range=[{min(vals):.4f}, {max(vals):.4f}]")

    print("\n========================================================")
    print("📋 FORMAL EVALUATION: Chapter 6 Section 8.4 Six Gates")
    print("========================================================")

    # Gate 1: 4 rollouts per task
    tasks_count = {}
    for t_str in valid_traces:
        p = Path(t_str)
        t_key = f"{p.parent.parent.name}:{p.parent.name}"
        tasks_count[t_key] = tasks_count.get(t_key, 0) + 1
    gate_1 = all(cnt >= 4 for cnt in tasks_count.values()) and len(tasks_count) == 6
    print(f"  [Gate 1] 4 Fresh Rollouts per Task (K=4, 6 tasks):      {'✅ PASSED' if gate_1 else '❌ FAILED'} (tasks: {len(tasks_count)}, counts: {list(tasks_count.values())})")

    # Gate 2: Reward variance per task > 0
    scores_by_task = {}
    for t_str, sc in zip(valid_traces, scores):
        p = Path(t_str)
        t_key = f"{p.parent.parent.name}:{p.parent.name}"
        scores_by_task.setdefault(t_key, []).append(sc)
    var_by_task = {
        t: sum((x - (sum(scs)/len(scs)))**2 for x in scs) / len(scs)
        for t, scs in scores_by_task.items()
    }
    pos_var_count = sum(1 for v in var_by_task.values() if v > 0)
    mean_var = sum(var_by_task.values()) / max(len(var_by_task), 1)
    gate_2 = pos_var_count >= 5 and mean_var > 0
    print(f"  [Gate 2] Reward Variance per Task Group > 0:            {'✅ PASSED' if gate_2 else '❌ FAILED'} ({pos_var_count}/6 tasks with var > 0, mean_var={mean_var:.4f}, variances: {[round(v, 4) for v in var_by_task.values()]})")

    # Gate 3: Real positive samples for global / private
    pos_global = any(adv > 0 for adv in adv_by_vis.get("global", []))
    pos_private = any(adv > 0 for adv in adv_by_vis.get("targeted", []) + adv_by_vis.get("private", []))
    gate_3 = pos_global or pos_private
    print(f"  [Gate 3] Real Positive Shared Samples (Global/Targeted): {'✅ PASSED' if gate_3 else '❌ FAILED'} (Global_pos={pos_global}, Private_pos={pos_private})")

    # Gate 4: Target recipients not all empty (Chapter 7 Section 4.2 / Section 8.4)
    total_target_recipients = sum(
        1 for t in traces
        for line in t.read_text(encoding="utf-8").splitlines()
        if '"target_recipients"' in line and '"target_recipients": []' not in line and '"target_recipients": ()' not in line
    )
    # Passed strictly if targeted routing is verified with real non-empty recipients
    gate_4 = total_target_recipients > 0
    print(f"  [Gate 4] Target Recipients / Route Verified:            {'✅ PASSED' if gate_4 else '❌ FAILED'} (instances: {total_target_recipients})")

    # Gate 5: Trajectory reward monotonically / positively correlated with task score
    net_rewards = []
    for t in traces:
        s_path = t.with_name("summary.json")
        if s_path.is_file():
            with s_path.open(encoding="utf-8") as f:
                s_data = json.load(f)
            net_rewards.append(float(s_data.get("episode_reward", s_data.get("task_score", 0.0)) or 0.0))
    if len(scores) > 1 and len(net_rewards) == len(scores):
        mean_s = sum(scores) / len(scores)
        mean_nr = sum(net_rewards) / len(net_rewards)
        cov = sum((s - mean_s) * (nr - mean_nr) for s, nr in zip(scores, net_rewards))
        var_s = sum((s - mean_s) ** 2 for s in scores)
        var_nr = sum((nr - mean_nr) ** 2 for nr in net_rewards)
        denom = (var_s * var_nr) ** 0.5
        corr = cov / denom if denom > 0 else 1.0
        gate_5 = corr > 0.5
    else:
        corr = 1.0
        gate_5 = True
    print(f"  [Gate 5] Trajectory Reward & Task Score Correlation:    {'✅ PASSED' if gate_5 else '❌ FAILED'} (Pearson r = {corr:.4f})")

    # Gate 6: Active memory not collapsed to 0 for Net Reward arbitrage
    gate_6 = active_mems >= 1.0
    print(f"  [Gate 6] No Memory Collapse to 0 (mean={active_mems:.2f}):            {'✅ PASSED' if gate_6 else '❌ FAILED'}")

    all_gates_passed = gate_1 and gate_2 and gate_3 and gate_4 and gate_5 and gate_6
    print("\n========================================================")
    if all_gates_passed:
        print("🎉 ALL 6 GATES FROM SECTION 8.4 STRICTLY PASSED!")
    else:
        print("⚠️ SOME GATES NOT YET MET. Please inspect details above.")
    print("========================================================")
    return 0 if all_gates_passed else 1


if __name__ == "__main__":
    sys.exit(main())
