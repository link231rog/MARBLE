#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Master Batch Runner (8 Concurrency per Baseline)
# Iterates through selected baselines, executing each with 8 parallel workers.
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

DEFAULT_BASELINES=(
    "ours_base"
    "no_memory"
    "global_add_all"
    "lts_style"
    "mem0_style"
    "amem_style"
    "memoryos_style"
    "memory_r1_style"
)

SPLIT="test"
CONCURRENCY=8
BASELINES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        -c|--concurrency)
            CONCURRENCY="$2"
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

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="runs/batch_c${CONCURRENCY}_${SPLIT}_${TIMESTAMP}"
mkdir -p "$OUT_ROOT"

echo "======================================================================"
echo "🚀 MARBLE Batch Benchmark Orchestrator"
echo "  Concurrency: $CONCURRENCY"
echo "  Split:       $SPLIT"
echo "  Baselines:   ${BASELINES[*]}"
echo "  Output Dir:  $OUT_ROOT"
echo "======================================================================"

for b in "${BASELINES[@]}"; do
    echo ""
    echo "######################################################################"
    echo ">>> [Batch Runner] Executing Baseline: $b (Split: $SPLIT, Concurrency: $CONCURRENCY)"
    echo "######################################################################"
    ./scripts/run_parallel_benchmark.sh \
        --baseline "$b" \
        --split "$SPLIT" \
        --concurrency "$CONCURRENCY" \
        --out "$OUT_ROOT/$b"
done

echo ""
echo "======================================================================"
echo "📊 Evaluating Full Batch Results Across Baselines..."
echo "======================================================================"
python3 -m marble.experiments.evaluate \
    --run-dir "$OUT_ROOT" \
    --manifest "configs/experiments/multiagentbench_stratified_frozen.json" \
    --split "$SPLIT" || true

echo "======================================================================"
echo "🎉 Batch Evaluation Finished! All results saved in: $OUT_ROOT"
echo "======================================================================"
