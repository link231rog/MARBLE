#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE 32-Concurrency Batch Benchmark Orchestrator
# Runs pairs of baselines concurrently (16 workers each = 32 total workers).
# Pair A: Port Offset 0 (DB Ports 54320-54327)
# Pair B: Port Offset 8 (DB Ports 54328-54335)
# Sandboxed Docker compose projects: marble_db_<baseline>_w<id>
# Multi-provider API rotation across SiliconFlow + OpenRouter + NVIDIA NIM.
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

UV="/Users/huangzixuan/.local/bin/uv"
MANIFEST="configs/experiments/multiagentbench_stratified_frozen.json"

DEFAULT_BASELINES=(
    "mem0_style"
    "amem_style"
    "memoryos_style"
    "memory_r1_style"
    "no_memory"
    "global_add_all"
    "lts_style"
    "single_agent"
)

SPLIT="test"
DRY_RUN=""
OUT_ROOT=""
BASELINES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift 1
            ;;
        -o|--out)
            OUT_ROOT="$2"
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
if [ -z "$OUT_ROOT" ]; then
    OUT_ROOT="runs/batch_32c_${SPLIT}_${TIMESTAMP}"
fi
mkdir -p "$OUT_ROOT"

echo "======================================================================"
echo "🚀 MARBLE 32-Concurrency Batch Runner"
echo "  Split:       $SPLIT"
echo "  Baselines:   ${BASELINES[*]}"
echo "  Output Dir:  $OUT_ROOT"
echo "  Strategy:    Concurrent Pairs (16 + 16 = 32 parallel workers)"
echo "======================================================================"

# Trap SIGINT/SIGTERM to kill running background jobs and clean up
cleanup_on_interrupt() {
    echo ""
    echo "⚠️ Interrupted! Terminating child jobs and cleaning up containers..."
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    docker ps -q --filter "name=marble_db_" | xargs -r docker rm -f >/dev/null 2>&1 || true
    exit 130
}
trap cleanup_on_interrupt SIGINT SIGTERM

num_baselines=${#BASELINES[@]}
idx=0

while [ $idx -lt $num_baselines ]; do
    b1="${BASELINES[$idx]}"
    b2=""
    if [ $((idx + 1)) -lt $num_baselines ]; then
        b2="${BASELINES[$((idx + 1))]}"
    fi

    if [ -n "$b2" ]; then
        echo ""
        echo "======================================================================"
        echo "🔥 [32-Concurrency Wave] Launching Pair in Parallel:"
        echo "   Baseline 1: $b1 (Port Offset 0, Ports 54320-54327, 16 workers)"
        echo "   Baseline 2: $b2 (Port Offset 8, Ports 54328-54335, 16 workers)"
        echo "   Total Simultaneous Active Workers: 32"
        echo "======================================================================"

        # Launch Baseline 1 (Port Offset 0)
        ./scripts/run_parallel_benchmark.sh \
            --baseline "$b1" \
            --split "$SPLIT" \
            --concurrency 16 \
            --port-offset 0 \
            $DRY_RUN \
            --out "$OUT_ROOT/$b1" 2>&1 | tee "$OUT_ROOT/${b1}_stream.log" &
        PID1=$!

        # Launch Baseline 2 (Port Offset 8)
        ./scripts/run_parallel_benchmark.sh \
            --baseline "$b2" \
            --split "$SPLIT" \
            --concurrency 16 \
            --port-offset 8 \
            $DRY_RUN \
            --out "$OUT_ROOT/$b2" 2>&1 | tee "$OUT_ROOT/${b2}_stream.log" &
        PID2=$!

        echo ">>> Both baselines ($b1, $b2) active! PIDs: $PID1, $PID2. Waiting for wave to finish..."
        
        # Wait for both and record exit codes
        exit_code1=0
        exit_code2=0
        wait "$PID1" || exit_code1=$?
        wait "$PID2" || exit_code2=$?

        echo ">>> Wave finished. $b1 exit code: $exit_code1, $b2 exit code: $exit_code2."
        if [ $exit_code1 -ne 0 ] || [ $exit_code2 -ne 0 ]; then
            echo "⚠️ Warning: One or both baselines encountered errors. Check logs."
        fi

        idx=$((idx + 2))
    else
        echo ""
        echo "======================================================================"
        echo "⚡ [16-Concurrency Single] Launching Final Baseline: $b1"
        echo "======================================================================"

        ./scripts/run_parallel_benchmark.sh \
            --baseline "$b1" \
            --split "$SPLIT" \
            --concurrency 16 \
            --port-offset 0 \
            $DRY_RUN \
            --out "$OUT_ROOT/$b1" 2>&1 | tee "$OUT_ROOT/${b1}_stream.log"

        idx=$((idx + 1))
    fi
done

echo ""
echo "======================================================================"
echo "📊 Evaluating Full Batch Results Across All Completed Baselines..."
echo "======================================================================"
$UV run python -m marble.experiments.evaluate \
    --run-dir "$OUT_ROOT" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" || true

echo "======================================================================"
echo "🎉 32-Concurrency Batch Evaluation Complete!"
echo "   All results stored in: $OUT_ROOT"
echo "======================================================================"
