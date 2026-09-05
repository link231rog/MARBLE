#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Round / Iteration Parameter Sensitivity Suite (T in {2, 4, 6, 8, 10})
# Evaluates how private memory emergence, context pollution, and task accuracy
# scale with interaction horizon T on high-difficulty long-horizon tasks.
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

UV="${UV:-$(command -v uv || echo "uv")}"
MANIFEST="configs/experiments/multiagentbench_hard_frozen.json"
SPLIT="test_hard"
CONCURRENCY="${CONCURRENCY:-4}"
CKPT="${1:-runs/ours_rl_v2_policy.json}"

cleanup_docker() {
    echo "Cleaning up lingering Docker containers..."
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
}

echo "======================================================================"
echo "🧪 MARBLE Iteration/Round Sensitivity Sweep Pipeline"
echo "  Manifest:    $MANIFEST"
echo "  Split:       $SPLIT"
echo "  Policy:      $CKPT"
echo "  Sweep T:     2, 4, 6, 8, 10"
echo "  Concurrency: $CONCURRENCY Workers"
echo "======================================================================"

cleanup_docker

is_run_completed() {
    local dir="$1"
    if [ -d "$dir" ]; then
        local count=$(find "$dir" -name "summary.json" 2>/dev/null | grep -v "evaluation_summary.json" | wc -l)
        if [ "$count" -ge 8 ]; then
            return 0
        fi
    fi
    return 1
}

SWEEP_ROUNDS=(2 4 6 8 10)

for T in "${SWEEP_ROUNDS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "⚡ [Iteration Sweep] Evaluating Horizon T = $T Rounds"
    echo "Time: $(date)"
    echo "======================================================================"

    OUT_DIR_OURS="runs/sensitivity_rounds_T${T}_ours_rl"
    OUT_DIR_GLOBAL="runs/sensitivity_rounds_T${T}_global_add_all"
    mkdir -p "$OUT_DIR_OURS" "$OUT_DIR_GLOBAL"

    # 1. Run MARBLE (ours_rl with Private/Global Governance)
    if is_run_completed "$OUT_DIR_OURS"; then
        echo ">>> [T=$T] MARBLE (ours_rl) already completed (8/8 tasks). Skipping..."
    else
        echo ">>> Running MARBLE (ours_rl) at T = $T..."
        ./scripts/run_parallel_benchmark.sh \
            --baseline ours_rl \
            --manifest "$MANIFEST" \
            --split "$SPLIT" \
            --concurrency "$CONCURRENCY" \
            --max-iterations "$T" \
            --controller-checkpoint "$CKPT" \
            --out "$OUT_DIR_OURS"
        cleanup_docker
        sleep 3
    fi

    # 2. Run GlobalAddAll (Unconstrained Global Broadcast Baseline)
    if is_run_completed "$OUT_DIR_GLOBAL"; then
        echo ">>> [T=$T] GlobalAddAll already completed (8/8 tasks). Skipping..."
    else
        echo ">>> Running GlobalAddAll (no privacy baseline) at T = $T..."
        ./scripts/run_parallel_benchmark.sh \
            --baseline global_add_all \
            --manifest "$MANIFEST" \
            --split "$SPLIT" \
            --concurrency "$CONCURRENCY" \
            --max-iterations "$T" \
            --out "$OUT_DIR_GLOBAL"
        cleanup_docker
        sleep 3
    fi

    echo "📊 Updating sensitivity analysis table after T=$T..."
    $UV run python -m marble.experiments.evaluate_round_sensitivity \
        --pattern "runs/sensitivity_rounds_T*" \
        --manifest "$MANIFEST" \
        --split "$SPLIT" \
        --out "runs/round_sensitivity_analysis_table.json" || true
done

echo ""
echo "======================================================================"
echo "🎉 ALL ITERATION SENSITIVITY RUNS COMPLETED!"
echo "======================================================================"

echo "📊 Aggregating round sensitivity results across horizons T..."
$UV run python -m marble.experiments.evaluate_round_sensitivity \
    --pattern "runs/sensitivity_rounds_T*" \
    --manifest "$MANIFEST" \
    --split "$SPLIT" \
    --out "runs/round_sensitivity_analysis_table.json"

echo "✅ All done! Results saved to runs/round_sensitivity_analysis_table.json"
