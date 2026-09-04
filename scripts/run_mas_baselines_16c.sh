#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE 16-Concurrency Runner for 2025-2026 Dedicated Multi-Agent Memory Baselines
# - g_memory_style  (NeurIPS 2025)
# - collabmem_style (ICML 2025/2026)
# - copper_style    (NeurIPS 2024/2025)
# - ours_rl_v2      (SFT Warm-start + RLVR Outcome Gating + SimPO Density Penalty)
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

UV="/Users/huangzixuan/.local/bin/uv"
SPLIT="test"

echo "======================================================================"
echo "⏳ Checking for existing ablation/benchmark runs..."
echo "======================================================================"
while pgrep -f "run_ablations_and_sensitivity.sh" >/dev/null 2>&1; do
    echo "[$(date '+%H:%M:%S')] Waiting for run_ablations_and_sensitivity.sh to complete..."
    sleep 30
done

cleanup_docker() {
    echo "Cleaning up lingering Docker containers..."
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
}

cleanup_docker

MAS_BASELINES=(
    "g_memory_style"
    "collabmem_style"
    "copper_style"
)

echo "======================================================================"
echo "🚀 Starting 2025-2026 Dedicated Multi-Agent Memory Baselines & SOTA Suite"
echo "  Baselines:   ${MAS_BASELINES[*]}"
echo "  Split:       $SPLIT"
echo "  Concurrency: 16 Workers"
echo "======================================================================"

TOTAL_START=$(date +%s)

for idx in "${!MAS_BASELINES[@]}"; do
    b="${MAS_BASELINES[$idx]}"
    STEP_NUM=$((idx + 1))
    echo ""
    echo "======================================================================"
    echo "[$STEP_NUM/${#MAS_BASELINES[@]}] ⚡ Launching MAS Baseline: '$b' (16 Concurrency, Split: $SPLIT)"
    echo "Time: $(date)"
    echo "======================================================================"
    OUT_DIR="runs/eval_${b}_${SPLIT}"
    mkdir -p "$OUT_DIR"

    ./scripts/run_parallel_benchmark.sh \
        --baseline "$b" \
        --split "$SPLIT" \
        --concurrency 16 \
        --out "$OUT_DIR"

    echo ">>> Baseline '$b' completed successfully!"
    cleanup_docker
    sleep 5
done

# Run Ours SOTA RL v2 with trained warmstart checkpoint
echo ""
echo "======================================================================"
echo "⚡ Launching Ours SOTA: 'ours_rl_v2' (16 Concurrency, Split: $SPLIT)"
echo "Time: $(date)"
echo "======================================================================"
OUT_DIR_RL2="runs/eval_ours_rl_v2_${SPLIT}"
mkdir -p "$OUT_DIR_RL2"

./scripts/run_parallel_benchmark.sh \
    --baseline ours_rl \
    --split "$SPLIT" \
    --concurrency 16 \
    --controller-checkpoint runs/ours_rl_v2_policy.json \
    --out "$OUT_DIR_RL2"

cleanup_docker
sleep 5

TOTAL_END=$(date +%s)
DURATION=$((TOTAL_END - TOTAL_START))

echo ""
echo "======================================================================"
echo "🎉 ALL MAS BASELINES & OURS_RL_V2 COMPLETED in ${DURATION}s!"
echo "======================================================================"
echo "📊 Aggregating Global Benchmark Table Across ALL Baselines..."
echo "======================================================================"

mkdir -p runs/eval_all_baselines_aggregated
for d in runs/eval_*_test; do
    bname="$(basename "$d")"
    if [ ! -e "runs/eval_all_baselines_aggregated/$bname" ]; then
        ln -s "../$bname" "runs/eval_all_baselines_aggregated/$bname" 2>/dev/null || cp -R "$d" "runs/eval_all_baselines_aggregated/$bname"
    fi
done

$UV run python -m marble.experiments.evaluate \
    --run-dir runs/eval_all_baselines_aggregated \
    | tee runs/final_paper_benchmark_table_expanded.json

echo ""
echo "======================================================================"
echo "✅ Comprehensive 14-Baseline Benchmark Table Ready!"
echo "   Output: runs/final_paper_benchmark_table_expanded.json"
echo "======================================================================"
