#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Master Chained Pipeline:
# Step 1: Run Table 1 Ours-RL evaluation on test_hard (24 tasks, concurrency 8)
# Step 2: Aggregate Table 1 summary
# Step 3: Run Parameter Scaling Suite across 0.8B, 2B, 9B (Base, SFT, RL)
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "========================================================================"
echo "⚡ [Pipeline Orchestrator] Table 1 Ours-RL -> Parameter Scaling (0.8B, 2B, 9B)"
echo "   Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

# Step 1: Execute Table 1 Ours-RL benchmark across 24 hard tasks
echo ""
echo "🚀 [Step 1/3] Executing Table 1 Ours-RL Evaluation (24 Test Hard Tasks)..."
mkdir -p runs/test_hard_ours_rl
./scripts/run_hard_benchmark_pool.sh \
    --baseline ours_rl \
    --split test_hard \
    --concurrency 8 \
    --controller-checkpoint runs/checkpoints/qwen_rl \
    --qwen-api-base "http://127.0.0.1:18000/v1" \
    --qwen-api-model qwen_rl \
    --out runs/test_hard_ours_rl \
    2>&1 | tee runs/test_hard_ours_rl_run.log

# Step 2: Generate Table 1 Summary
echo ""
echo "========================================================================"
echo "📊 [Step 2/3] Generating Master Table 1 Ours-RL Report..."
echo "========================================================================"
./.venv/bin/python -c "
import json, glob
summaries = []
for p in sorted(glob.glob('runs/test_hard_ours_rl/**/summary.json', recursive=True)):
    with open(p) as f:
        summaries.append(json.load(f))
if summaries:
    n = len(summaries)
    succ = sum(1 for s in summaries if s.get('task_success', False))
    score = sum(s.get('task_score', 0.0) for s in summaries) / n
    act_mem = sum(s.get('memory_metrics', {}).get('active_memory_count', s.get('active_memory_count', 0)) for s in summaries) / n
    cross_reads = sum(s.get('memory_metrics', {}).get('cross_agent_reads', s.get('cross_agent_reads', 0)) for s in summaries) / n
    prod_reads = sum(s.get('memory_metrics', {}).get('productive_cross_reads', s.get('productive_cross_reads', 0)) for s in summaries) / n
    prod_rate = sum(s.get('memory_metrics', {}).get('productive_cross_read_rate', s.get('productive_cross_read_rate', 0.0)) for s in summaries) / n
    neg_trans = sum(s.get('memory_metrics', {}).get('negative_transfer', s.get('negative_transfer', 0.0)) for s in summaries) / n
    print(f'=== TABLE 1 OURS-RL RESULTS (N={n}) ===')
    print(f'Success Rate:               {succ/n*100:.1f}%')
    print(f'Mean Task Score:            {score:.3f}')
    print(f'Active Memories:            {act_mem:.2f}')
    print(f'Cross-Agent Reads:          {cross_reads:.2f}')
    print(f'Productive Cross Reads:     {prod_reads:.2f} ({prod_rate*100:.1f}%)')
    print(f'Negative Transfer Rate:     {neg_trans*100:.1f}%')
" > runs/test_hard_ours_rl/table1_summary.txt 2>&1 || true

cat runs/test_hard_ours_rl/table1_summary.txt || true

# Step 3: Automatically Launch Parameter Scaling Analysis (0.8B, 2B, 9B)
echo ""
echo "========================================================================"
echo "🚀 [Step 3/3] Automatically Launching Parameter Scaling Analysis..."
echo "   Scales:  0.8b 2b 9b"
echo "   Methods: ours_base ours_sft ours_rl"
echo "========================================================================"

mkdir -p runs/controller_scaling_analysis
./scripts/run_controller_scaling_analysis.sh \
    --scales "0.8b 2b 9b" \
    --methods "ours_base ours_sft ours_rl" \
    --seed 42 \
    --concurrency 6 \
    > runs/controller_scaling_analysis/scaling_suite_0.8_2_9.log 2>&1

echo "========================================================================"
echo "🎉 ALL EXPERIMENTAL STAGES COMPLETED!"
echo "   End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
