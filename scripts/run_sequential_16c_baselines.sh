#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Sequential 16-Concurrency Baseline Benchmark Runner
# Iterates through remaining baselines one by one.
# Runs each baseline across 16 parallel workers (8 DB + 8 Research).
# Ensures clean isolation, zero GPU overhead, and low local load.
# Automatically aggregates final results across ALL baselines,
# then seamlessly auto-triggers the Ablation & Sensitivity experiment suite!
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

UV="/Users/huangzixuan/.local/bin/uv"

# Note: mem0_style is already 100% completed in runs/eval_mem0_style_test
DEFAULT_BASELINES=(
    "amem_style"
    "memoryos_style"
    "memory_r1_style"
    "no_memory"
    "global_add_all"
    "lts_style"
    "single_agent"
)

SPLIT="test"
BASELINES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        -b|--baseline)
            BASELINES+=("$2")
            shift 2
            ;;
        *)
            BASELINES+=("$1")
            shift 1
            ;;
    esac
done

if [ ${#BASELINES[@]} -eq 0 ]; then
    BASELINES=("${DEFAULT_BASELINES[@]}")
fi

echo "======================================================================"
echo "🚀 Starting MARBLE 16-Concurrency Sequential Sweep"
echo "  Split:     $SPLIT"
echo "  Baselines: ${BASELINES[*]}"
echo "  Total:     ${#BASELINES[@]} baselines"
echo "======================================================================"

cleanup_docker() {
    echo "Cleaning up lingering Docker containers..."
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
}

cleanup_docker

TOTAL_START=$(date +%s)

for idx in "${!BASELINES[@]}"; do
    b="${BASELINES[$idx]}"
    STEP_NUM=$((idx + 1))
    echo ""
    echo "======================================================================"
    echo "[$STEP_NUM/${#BASELINES[@]}] ⚡ Launching Baseline: '$b' (16 Concurrency, Split: $SPLIT)"
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

TOTAL_END=$(date +%s)
DURATION=$((TOTAL_END - TOTAL_START))

echo ""
echo "======================================================================"
echo "🎉 ALL BASELINES SUCCESSFULLY COMPLETED in ${DURATION}s!"
echo "======================================================================"
echo "📊 Aggregating Global Benchmark Table Across ALL 11 Baselines..."
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
    | tee runs/final_paper_benchmark_table.json

echo ""
echo "======================================================================"
echo "✅ Complete Paper Benchmark Run & Evaluation Table Ready!"
echo "   Output: runs/final_paper_benchmark_table.json"
echo "======================================================================"

echo ""
echo "======================================================================"
echo "🚀 [AUTO-CHAIN] Seamlessly launching Ablations & Sensitivity Studies..."
echo "======================================================================"
./scripts/run_ablations_and_sensitivity.sh
