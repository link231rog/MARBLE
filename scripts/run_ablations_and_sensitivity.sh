#!/bin/bash
set -euo pipefail

# ==============================================================================
# MARBLE Ablation Studies & Hyperparameter Sensitivity Pipeline
# Executes core ablation variants and sensitivity sweeps on ours_rl.
# Automatically aggregates results into final comparison table.
# ==============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

UV="${UV:-$(command -v uv || echo "uv")}"
SPLIT="test"
CKPT="runs/ours_rl_policy.json"

cleanup_docker() {
    echo "Cleaning up lingering Docker containers..."
    docker ps -q | xargs -r docker rm -f >/dev/null 2>&1 || true
}

echo "======================================================================"
echo "🧪 Starting MARBLE Ablation & Sensitivity Experiment Suite"
echo "  Policy Checkpoint: $CKPT"
echo "  Split:             $SPLIT"
echo "  Concurrency:       16 Workers"
echo "======================================================================"

cleanup_docker

# 1. State Update Ablation (NoSupersede: evaluates importance of memory overwriting)
echo ""
echo "======================================================================"
echo "[Ablation 1/4] ⚡ State Update Off (state_update:off)"
echo "Time: $(date)"
echo "======================================================================"
./scripts/run_parallel_benchmark.sh \
    --baseline ours_rl \
    --split "$SPLIT" \
    --concurrency 16 \
    --controller-checkpoint "$CKPT" \
    --ablation state_update:off \
    --out runs/eval_ablation_state_update_off_test
cleanup_docker
sleep 5

# 2. Retrieval Ablation (NoRetrieval: max_cards=0)
echo ""
echo "======================================================================"
echo "[Ablation 2/4] ⚡ Retrieval None (retrieval:none)"
echo "Time: $(date)"
echo "======================================================================"
./scripts/run_parallel_benchmark.sh \
    --baseline ours_rl \
    --split "$SPLIT" \
    --concurrency 16 \
    --controller-checkpoint "$CKPT" \
    --ablation retrieval:none \
    --out runs/eval_ablation_retrieval_none_test
cleanup_docker
sleep 5

# 3. Sensitivity: Lambda = 0.0 (No penalty for memory expansion)
echo ""
echo "======================================================================"
echo "[Sensitivity 3/4] ⚡ Memory Penalty Lambda = 0.0"
echo "Time: $(date)"
echo "======================================================================"
./scripts/run_parallel_benchmark.sh \
    --baseline ours_rl \
    --split "$SPLIT" \
    --concurrency 16 \
    --controller-checkpoint "$CKPT" \
    --lambda 0.0 \
    --out runs/eval_sensitivity_lambda_0_test
cleanup_docker
sleep 5

# 4. Sensitivity: Lambda = 0.30 (Heavy penalty for memory expansion)
echo ""
echo "======================================================================"
echo "[Sensitivity 4/4] ⚡ Memory Penalty Lambda = 0.30"
echo "Time: $(date)"
echo "======================================================================"
./scripts/run_parallel_benchmark.sh \
    --baseline ours_rl \
    --split "$SPLIT" \
    --concurrency 16 \
    --controller-checkpoint "$CKPT" \
    --lambda 0.30 \
    --out runs/eval_sensitivity_lambda_030_test
cleanup_docker
sleep 5

echo ""
echo "======================================================================"
echo "🎉 ALL ABLATION & SENSITIVITY RUNS COMPLETED!"
echo "======================================================================"
echo "📊 Aggregating Ablation & Sensitivity Report..."
echo "======================================================================"

mkdir -p runs/eval_ablations_aggregated
for d in runs/eval_ablation_* runs/eval_sensitivity_*; do
    bname="$(basename "$d")"
    if [ ! -e "runs/eval_ablations_aggregated/$bname" ]; then
        ln -s "../$bname" "runs/eval_ablations_aggregated/$bname" 2>/dev/null || cp -R "$d" "runs/eval_ablations_aggregated/$bname"
    fi
done

$UV run python -m marble.experiments.evaluate \
    --run-dir runs/eval_ablations_aggregated \
    | tee runs/final_ablation_and_sensitivity_table.json

echo ""
echo "======================================================================"
echo "✅ Ablation & Sensitivity Table Ready at runs/final_ablation_and_sensitivity_table.json"
echo "======================================================================"
