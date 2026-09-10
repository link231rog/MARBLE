#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Master Table 1: Classical Baselines Evaluation Suite (Decoupled)
# Executes ONLY the 9 classical/mature baselines across 24 Hard Episodes:
#   1. single_agent
#   2. no_memory
#   3. global_add_all
#   4. mem0_style
#   5. lts_style
#   6. memory_r1_style
#   7. g_memory_style
#   8. collabmem_style
#   9. copper_style
#
# Worker Model: openai/gpt-oss-20b (via NVIDIA NIM)
# Note: Completely decoupled from 'ours_*' methods so classical numbers are
# frozen and immune to fine-tuning/training cycles.
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

CONCURRENCY="${CONCURRENCY:-24}"
SPLIT="${SPLIT:-test_hard}"
MANIFEST="${MANIFEST:-configs/experiments/multiagentbench_hard_frozen.json}"

# The 9 canonical classical baselines for Table 1
if [ $# -gt 0 ]; then
    BASELINES=("$@")
else
    BASELINES=(
        "single_agent"
        "no_memory"
        "global_add_all"
        "mem0_style"
        "lts_style"
        "memory_r1_style"
        "g_memory_style"
        "collabmem_style"
        "copper_style"
    )
fi

echo "======================================================================"
echo "🚀 MARBLE Classical Baselines Benchmark Suite (Decoupled)"
echo "  Split:       $SPLIT (24 Episodes: 8 DB + 8 Research + 8 Coding)"
echo "  Manifest:    $MANIFEST"
echo "  Concurrency: $CONCURRENCY"
echo "  Worker Model: ${MARBLE_WORKER_MODEL:-openai/gpt-oss-20b}"
echo "  Baselines (${#BASELINES[@]}): ${BASELINES[*]}"
echo "======================================================================"

for i in "${!BASELINES[@]}"; do
    b="${BASELINES[$i]}"
    echo ""
    echo "======================================================================"
    echo "[$((i+1))/${#BASELINES[@]}] ⚡ Checking Classical Baseline: $b"
    echo "Time: $(date)"
    echo "======================================================================"

    # Check if this baseline already completed 24 episodes in runs/ with the current worker model
    done_run=""
    target_worker="gpt-oss-20b"
    for r in runs/hard_${b}_${SPLIT}_c${CONCURRENCY}_*; do
        if [ -d "$r" ] && [ $(find "$r" -name "summary.json" 2>/dev/null | wc -l) -ge 24 ]; then
            first_sum=$(find "$r" -name "summary.json" 2>/dev/null | head -n 1)
            if [ -n "$first_sum" ] && grep -q "$target_worker" "$first_sum"; then
                done_run="$r"
                break
            fi
        fi
    done

    if [ -n "$done_run" ]; then
        echo "✅ Baseline '$b' already has 24/24 completed episodes in $done_run (matching $target_worker). Skipping to next!"
        continue
    fi

    echo "⚡ Launching Classical Baseline: $b..."
    ./scripts/run_hard_benchmark.sh \
        --baseline "$b" \
        --split "$SPLIT" \
        --manifest "$MANIFEST" \
        --concurrency "$CONCURRENCY"

    # Clean lingering docker containers and anonymous volumes between baselines
    [ -n "$(docker ps -q)" ] && docker rm -f $(docker ps -q) >/dev/null 2>&1 || true
    docker volume prune -f >/dev/null 2>&1 || true
    sleep 6
done

echo ""
echo "======================================================================"
echo "🔍 Running Final Auto-Self-Healing Sweep Across All Baselines..."
echo "======================================================================"
./scripts/backfill_missing_tasks.sh "$MANIFEST" "$SPLIT" || true

echo ""
echo "======================================================================"
echo "🎉 All Classical Baselines Complete! Aggregating Results..."
echo "Time: $(date)"
echo "======================================================================"

python3 scripts/aggregate_hard_master_table.py \
    --run-dir runs/ \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --out-json "runs/master_table_classical_baselines.json"

echo "======================================================================"
echo "✅ Results saved to runs/master_table_classical_baselines.json"
echo "======================================================================"
